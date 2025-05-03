from __future__ import annotations
from typing import List, Tuple, Dict, Union, Optional
from argparse import ArgumentParser
import torch

from separ.data.struct import Arc, Bracket 
from separ.utils import flatten, Config, bar 
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc
from separ.models.dep.parser import DependencySLParser


class BracketDependencyParser(DependencySLParser):
    """Bracket Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-bracket'
    PARAMS = ['k']
    
    class Labeler(DependencySLParser.Labeler):
        """Bracketing encoding for k-planar dependency trees"""
        DEFAULT = ''
        BRACKETS = ['<', '\\', '/', '>']
        
        def __init__(self, k: int = 2):
            self.k  = k
            
        def __repr__(self) -> str:
            return f'BracketDependencyLabeler(k={self.k})'
        
        def encode(self, tree: CoNLL.Tree) -> Tuple[List[str], List[str]]:
            brackets = [[] for _ in range(len(tree))]
            for p, plane in tree.relaxed_planes.items():
                if p >= self.k:
                    break 
                self.encode_plane(plane, p, brackets)
            brackets = [''.join(map(repr, sorted(bracket))) for bracket in brackets]
            return brackets, tree.DEPREL 
            
        def encode_plane(self, arcs: List[Arc], p: int, brackets: List[List[Bracket]]) -> List[Bracket]:
            for arc in arcs:
                if arc.side == 1:
                    brackets[arc.DEP-1].append(Bracket('>', p))
                    if arc.HEAD > 0:
                        brackets[arc.HEAD-1].append(Bracket('/', p))
                else:
                    brackets[arc.DEP-1].append(Bracket('<', p))
                    brackets[arc.HEAD-1].append(Bracket('\\', p))
            return brackets 
            
        def decode(self, brackets: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            brackets = list(map(Bracket.from_string, brackets))
            adjacent = torch.zeros(len(brackets) + 1, len(brackets) + 1, dtype=torch.bool)
            well_formed = True 
            for p in range(self.k):
                well_formed &= self.decode_plane(brackets, p, adjacent)
            return self.postprocess(adjacent, rels), well_formed
        
        def decode_plane(self, brackets: List[List[Bracket]], p: int, adjacent: torch.Tensor) -> bool:
            """
            Decodes the brackets of an specific plane.

            Args:
                brackets (List[List[Bracket]]): List of brackets of the same plane.
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
                    
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        k: int, 
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.k = k
        self.lab = self.Labeler(k)

    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser = super(BracketDependencyParser, cls).add_arguments(argparser)
        argparser.add_argument('-k', type=int, help='Number of planes')
        return argparser
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.BRACKET, tree.REL = self.lab.encode(tree)
            tree.transformed = True 
        return tree 
            
    def _pred(self, tree: CoNLL.Tree, bracket_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(self.BRACKET.decode(bracket_pred), self.REL.decode(rel_pred))
        return tree.rebuild_from_arcs(rec), well_formed
    
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        k: int = 2,
        device: int = 0,
        **_
    ) -> BracketDependencyParser:
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
            
        bracket_tkz, rel_tkz = TargetTokenizer('BRACKET'), TargetTokenizer('REL')
        labeler = cls.Labeler(k)
        brackets, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bracket_tkz.train(brackets)
        rel_tkz.train(rels)
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        return cls(input_tkzs, [bracket_tkz, rel_tkz], [enc_conf, *in_confs, bracket_tkz.conf, rel_conf], k, device)
