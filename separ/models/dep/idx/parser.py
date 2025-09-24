from __future__ import annotations 
from argparse import ArgumentParser
import torch

from separ.data import CoNLL
from separ.models.dep.sl import DependencySLParser


class IndexDependencyParser(DependencySLParser):
    """Index Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)"""
    NAME = 'dep-idx'
    
    class Labeler(DependencySLParser.Labeler):
        PARAMS = {'rel'}
        NAMES = ['INDEX', 'REL']

        def __init__(self, rel: bool, **_):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexDependencyLabeler(rel={self.rel})'
        
        @property
        def NAME(self) -> str:
            return f'dep-rel' if self.rel else 'dep-abs'
        
        @property 
        def NAMES(self) -> list[str]:
            if self.rel:
                return ['INDEX(R)', 'DEPREL']
            else:
                return ['INDEX(A)', 'DEPREL']
            
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
                
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    