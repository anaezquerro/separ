from __future__ import annotations 
from argparse import ArgumentParser
import torch, os


from separ.utils import Config, flatten, bar
from separ.data import CoNLL, InputTokenizer, PretrainedTokenizer, CharacterTokenizer, TargetTokenizer , Arc
from separ.models.dep.parser import DependencySLParser


class IndexDependencyParser(DependencySLParser):
    """Index Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)"""
    NAME = 'dep-idx'
    PARAMS = ['rel']
    
    class Labeler(DependencySLParser.Labeler):

        def __init__(self, rel: bool = False):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexDependencyLabeler(rel={self.rel})'
            
        def encode(self, graph: CoNLL.Tree) -> tuple[list[str], list[str]]:
            indexes, rels = ['' for _ in range(len(graph))], ['' for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1] = str(arc.HEAD - (arc.DEP*self.rel))
                rels[arc.DEP-1] = arc.REL 
            return indexes, rels 
        
        def decode(self, indexes: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            n = len(indexes)
            adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
            well_formed = True
            for idep, index in enumerate(indexes):
                dep = idep + 1
                head = int(index)+dep*self.rel 
                if head not in range(0, n+1) or not self.is_valid(adjacent, dep, head):
                    well_formed = False 
                else:
                    adjacent[dep, head] = True 
            return self.postprocess(adjacent, rels), well_formed and (n == adjacent.sum())
        
    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        rel: bool,
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.rel = rel
        self.lab = self.Labeler(rel)
            
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(IndexDependencyParser, cls).add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        """Transformation step of one tree for batching."""
        if not tree.transformed:
            tree.INDEX, tree.REL = self.lab.encode(tree)
            tree.transformed = True 
        return tree 
    
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        rel: bool = False,
        device: int = 0,
        **_
    ) -> IndexDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)

            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
            
        index_tkz, rel_tkz = TargetTokenizer('INDEX'), TargetTokenizer('REL')
        labeler = cls.Labeler(rel=rel)
        indexes, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        index_tkz.train(indexes)
        rel_tkz.train(rels)
        return cls(input_tkzs, [index_tkz, rel_tkz], [enc_conf, *in_confs, index_tkz.conf, rel_tkz.conf], rel, device)
        
        


   