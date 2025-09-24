from __future__ import annotations
from argparse import ArgumentParser

from separ.data import SDP, TargetTokenizer, EnhancedCoNLL, Arc, Graph 
from separ.utils import Config
from separ.models.grp.sl import GraphSLParser


class IndexGraphParser(GraphSLParser):
    """
    Index Graph Parser from [Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/).
    """
    NAME = 'grp-idx'
    
    class Labeler(GraphSLParser.Labeler):
        PARAMS = {'rel'}
        
        def __init__(self, rel: bool, **_):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexSemanticLabeler(rel={self.rel})'
        
        @property 
        def NAMES(self) -> list[str]:
            if self.rel:
                return ['INDEX(R)', 'REL']
            else:
                return ['INDEX(A)', 'REL']
            
        @property
        def NAME(self) -> str:
            return f'grp-rel' if self.rel else 'grp-abs'
        
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            indexes = [[] for _ in range(len(graph))]
            rels = [[] for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1].append(arc.HEAD - (arc.DEP*self.rel))
                rels[arc.DEP-1].append(arc.REL)
            indexes = ['' if len(index) == 0 else self.SEP.join(map(str, sorted(index))) for index in indexes]
            return indexes, [self.SEP.join(rel) for rel in rels]
        
        def decode(self, indexes: list[str], rels: list[str] | None = None) -> tuple[list[Arc], bool]:
            n, arcs, well_formed = len(indexes), [], True
            rels = [[] for _ in indexes] if rels is None else [rel.split(self.SEP) for rel in rels]
            for idep, index in enumerate(indexes):
                if index == '':
                    continue 
                for idx in index.split(self.SEP):
                    head = int(idx)+(idep+1)*self.rel 
                    if head in range(n+1):
                        arcs.append(Arc(head, idep+1, rels[idep].pop(0) if len(rels[idep]) > 0 else self.REL))
                    else:
                        well_formed = False
            return sorted(arcs), well_formed
        
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = GraphSLParser.add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    
        