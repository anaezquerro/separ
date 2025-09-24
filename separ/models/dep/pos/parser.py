from __future__ import annotations 
import torch 

from separ.models.dep.sl import DependencySLParser
from separ.utils import Config, flatten, bar
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc, forms_cycles

class PoSDependencyParser(DependencySLParser):
    """PoS-tag Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-pos'
    
    class Labeler(DependencySLParser.Labeler):
        SEP = '$'
        DEFAULT = 'root$-1'
        NAMES = ['INDEX(P)', 'UPOS', 'REL']
        NAME = 'dep-pos'

        def __init__(self, **_):
            pass 
        
        def __repr__(self) -> str:
            return f'PoSDependencyLabeler()'
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str], list[str]]:
            indexes, rels = ['' for _ in range(len(tree))], ['' for _ in range(len(tree))]
            tags = ['<bos>'] + tree.UPOS
            for arc in tree.arcs:
                pos = tags[arc.HEAD]
                if arc.side == -1: # left arc 
                    index = sum(tags[i] == pos for i in range(arc.DEP+1, arc.HEAD+1))
                else:
                    index = -sum(tags[i] == pos for i in range(arc.HEAD, arc.DEP))
                indexes[arc.DEP-1] = f'{pos}{self.SEP}{index}'
                rels[arc.DEP-1] = arc.REL 
            return indexes, list(tree.UPOS), rels
        
        def decode(self, indexes: list[str], tags: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            n, tags = len(indexes), ['<bos>'] + tags
            adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
            well_formed = True 
            for idep, label in enumerate(indexes):
                dep = idep + 1
                pos, index = label.split(self.SEP)
                index = int(index)
                if index > 0: # head is at right 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(dep+1, n+1)]
                    index -= 1
                    index = min(index, len(candidates)-1)
                else: # head is at left 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(0, dep)]
                    index = max(-len(candidates), index)
                if len(candidates) == 0 or not self.is_valid(adjacent, dep, candidates[index]):
                    well_formed = False
                    if not adjacent[:, 0].any().item():
                        head = 0
                    else:
                        head = [h for h in set(range(1,n+1)) if not forms_cycles(adjacent, dep, h) and h != dep].pop(0)
                else:
                    head = candidates[index]
                adjacent[dep, head] = True 
            return self.postprocess(adjacent, rels), well_formed and (n == adjacent.sum())

    @classmethod 
    def build(cls, **_) -> PoSDependencyParser:
        _['tag_conf'] = None
        return super().build(**_)