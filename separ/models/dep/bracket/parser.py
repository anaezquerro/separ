from __future__ import annotations
from argparse import ArgumentParser
import torch

from separ.data.struct import Arc, Bracket 
from separ.utils import flatten, Config, bar 
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc
from separ.models.dep.sl import DependencySLParser


class BracketDependencyParser(DependencySLParser):
    """Bracket Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-bracket'
    
    class Labeler(DependencySLParser.Labeler):
        """Bracketing encoding for k-planar dependency trees"""
        DEFAULT = ''
        BRACKETS = ['<', '\\', '/', '>']
        PARAMS = {'k'}
        
        def __init__(self, k: int, **_):
            self.k  = k
            
        def __repr__(self) -> str:
            return f'BracketDependencyLabeler(k={self.k})'
        
        @property
        def NAMES(self) -> list[str]:
            return [f'BRACKET({self.k})', 'DEPREL']
        
        @property
        def NAME(self) -> str:
            return f'dep-br({self.k})'
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str]]:
            brackets = [[] for _ in range(len(tree))]
            for p, plane in tree.relaxed_planes.items():
                if p >= self.k:
                    break 
                self.encode_plane(plane, p, brackets)
            brackets = [''.join(map(repr, sorted(bracket))) for bracket in brackets]
            return brackets, tree.DEPREL 
            
        def encode_plane(self, arcs: list[Arc], p: int, brackets: list[list[Bracket]]) -> list[Bracket]:
            for arc in arcs:
                if arc.side == 1:
                    brackets[arc.DEP-1].append(Bracket('>', p))
                    if arc.HEAD > 0:
                        brackets[arc.HEAD-1].append(Bracket('/', p))
                else:
                    brackets[arc.DEP-1].append(Bracket('<', p))
                    brackets[arc.HEAD-1].append(Bracket('\\', p))
            return brackets 
            
        def decode(self, brackets: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            brackets = list(map(Bracket.from_string, brackets))
            adjacent = torch.zeros(len(brackets) + 1, len(brackets) + 1, dtype=torch.bool)
            well_formed = True 
            for p in range(self.k):
                well_formed &= self.decode_plane(brackets, p, adjacent)
            return self.postprocess(adjacent, rels), well_formed
        
        def decode_plane(self, brackets: list[list[Bracket]], p: int, adjacent: torch.Tensor) -> bool:
            """
            Decodes the brackets of an specific plane.

            Args:
                brackets (list[list[Bracket]]): List of brackets of the same plane.
                p (int): Current plane.
                adjacent (torch.Tensor): Adjacent matrix.

            Returns:
                torch.Tensor: Adjacency matrix.
                bool: Whether the bracket sequence if well-formed.
            """
            right, left = [0] if p == 0 else [], []
            well_formed = True 
            for i, _brackets in enumerate(brackets):
                for bracket in _brackets:
                    if bracket.p != p:
                        continue 
                    if bracket.is_opening(): # opening symbol
                        if bracket.is_left():
                            left.append(i+1)
                        else:
                            right.append(i+1)
                    else:
                        if bracket.is_right():
                            if len(right) > 0 and self.is_valid(adjacent, i+1, right[-1]):
                                adjacent[i+1, right.pop(-1)] = True 
                            else:
                                well_formed = False 
                        else:
                            if len(left) > 0 and self.is_valid(adjacent, left[-1], i+1):
                                adjacent[left.pop(-1), i+1] = True 
                            else:
                                well_formed = False 
            return well_formed and len(right) == len(left) == 0

        def test(self, tree: CoNLL.Tree) -> bool:
            return super().test(tree.relaxed_planarize(self.k))

    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser.add_argument('-k', type=int, help='Number of planes')
        return argparser
