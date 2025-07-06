from __future__ import annotations 
import torch 

from separ.models.dep.parser import DependencySLParser
from separ.utils import Config, flatten, bar
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc, forms_cycles

class PoSDependencyParser(DependencySLParser):
    """PoS-tag Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-pos'
    
    class Labeler(DependencySLParser.Labeler):
        SEP = '$'
        DEFAULT = 'root$-1'
        
        def __init__(self):
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
            return indexes, rels, list(tree.UPOS)
        
        def decode(self, indexes: list[str], rels: list[str], tags: list[str]) -> tuple[list[Arc], bool]:
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

    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.lab = self.Labeler()
        
    @classmethod
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.INDEX, tree.REL, tree.TAG = self.lab.encode(tree)
            tree.transformed = True 
           
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        char_conf: Config | None = None,
        device: int = 0,
        **_
    ) -> PoSDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            word_tkz = InputTokenizer('WORD', 'FORM')
            word_tkz.train(data)
            word_conf.update(word_tkz.conf)
            input_tkzs = [word_tkz]
            
            if char_conf is not None:
                char_tkz = CharacterTokenizer('CHAR', 'FORM')
                char_tkz.train(data)
                char_conf.update(char_tkz.conf)
                input_tkzs.append(char_tkz)
                
            in_confs = [word_conf, None, char_conf]
            
        index_tkz, rel_tkz, tag_tkz = TargetTokenizer('INDEX'), TargetTokenizer('REL'), TargetTokenizer('TAG')
        labeler = cls.Labeler()
        indexes, rels, tags = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        index_tkz.train(indexes)
        rel_tkz.train(rels)
        tag_tkz.train(tags)
        return cls(input_tkzs, [index_tkz, rel_tkz, tag_tkz], [enc_conf, *in_confs, index_tkz.conf, rel_tkz.conf, tag_tkz.conf], device)
        
            