from __future__ import annotations
from argparse import ArgumentParser
import torch 

from separ.data import CoNLL, Arc, deprojectivize
from separ.models.dep.sl import DependencySLParser

class Bit4DependencyParser(DependencySLParser):
    """4-bit Dependency Parser from [Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)."""
    NAME = 'dep-bit4'
    
    class Labeler(DependencySLParser.Labeler):
        """4-bit encoding for projective dependency trees.
        
        - b0: word is a left dependant (0) or a right dependant (1) in the plane.
        - b1: word is the farthest dependant in the plane.
        - b2: word has left dependents in the plane.
        - b3: word has right dependents in the plane.
        """
        NUM_BITS = 4
        DEFAULT = '0000'
        PARAMS = {'proj'}
        
        def __init__(self, proj: str | None, **_):
            self.proj = proj
            
        def __repr__(self):
            return f'Bit4DependencyLabeler(proj={self.proj})'
        
        @property
        def NAME(self) -> str:
            abbr = ''.join(t[0] for t in self.proj.split('+'))
            return f'dep-4bit({abbr})'
        
        @property
        def NAMES(self) -> list[str]:
            if self.proj:
                abbr = ''.join(t[0] for t in self.proj.split('+'))
                return ['4BIT', f'DEPREL({abbr})']
            else:
                return ['4BIT', 'DEPREL']
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str]]:
            if self.proj:
                tree = tree.projectivize(self.proj)
            bits = [[False for _ in range(self.NUM_BITS)] for _ in range(len(tree))]
            for idep, head in enumerate(tree.HEAD):
                dep = idep+1
                bits[idep][0] = head < dep
                bits[idep][1] = (tree.ADJACENT[:, head].nonzero().max().item() == dep) if head < dep \
                    else (tree.ADJACENT[:, head].nonzero().min().item() == dep)
                bits[idep][2] = tree.ADJACENT[:dep, dep].any().item()
                bits[idep][3] = tree.ADJACENT[dep:, dep].any().item()
            bits = [''.join(map(str, map(int, label))) for label in bits]
            return bits, tree.DEPREL
        
        def decode(self, bits: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            left, right = [], [0]
            well_formed = True 
            adjacent = torch.zeros(len(bits)+1, len(bits)+1, dtype=torch.bool)
            for idep, label in enumerate(bits):
                b0, b1, b2, b3 = map(bool, map(int, label))
                if b0:
                    if len(right) > 0 and self.is_valid(adjacent, idep+1, right[-1]):
                        adjacent[idep+1, right[-1]] = True
                        if b1: # farthest dependant 
                            right.pop(-1)
                    else:
                        well_formed = False
                if b2:
                    last = False 
                    while len(left) > 0 and not last:
                        if self.is_valid(adjacent, left[-1][0], idep+1):
                            dep, last = left.pop(-1)
                            adjacent[dep, idep+1] = True
                        else:
                            well_formed = False
                            break
                if not b0:
                    left.append((idep+1, b1))
                if b3:
                    right.append(idep+1)
            well_formed = well_formed and (len(right + left) == 0) and (adjacent.sum() == len(bits))
            arcs = self.postprocess(adjacent, rels)
            if self.proj:
                arcs = deprojectivize(arcs, self.proj)
            return arcs, well_formed 
        
        def test(self, tree: CoNLL.Tree) -> bool:
            return super().test(tree.projectivize(mode='head'))
        
    def __init__(self, proj: str | None, **kwargs):
        super().__init__(**kwargs)
        self.lab = self.Labeler(proj)
        self.proj = proj 
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser.add_argument('--proj', default=None, type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 

