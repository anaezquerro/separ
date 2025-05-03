from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Tuple, Union, Optional
import torch 

from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc 
from separ.utils import flatten, Config, bar 
from separ.models.dep.parser import DependencySLParser

class Bit4DependencyParser(DependencySLParser):
    """4-bit Dependency Parser from [Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)."""
    NAME = 'dep-bit4'
    PARAMS = ['proj']
    
    class Labeler(DependencySLParser.Labeler):
        """4-bit encoding for projective dependency trees.
        
        - b0: word is a left dependant (0) or a right dependant (1) in the plane.
        - b1: word is the farthest dependant in the plane.
        - b2: word has left dependants in the plane.
        - b3: word has right dependants in the plane.
        """
        NUM_BITS = 4
        DEFAULT = '0000'
        
        def __init__(self, proj: Optional[str] = None):
            self.proj = proj
            
        def __repr__(self):
            return f'Bit4DependencyLabeler(proj={self.proj})'
        
        def encode(self, tree: CoNLL.Tree) -> Tuple[List[str], List[str]]:
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
            return [''.join(map(str, map(int, label))) for label in bits], list(tree.DEPREL)
        
        def decode(self, bits: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
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
            return self.postprocess(adjacent, rels), well_formed 
        
        def test(self, graph: CoNLL.Tree) -> bool:
            return super().test(graph.planarize(1))
    
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        proj: Optional[str],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.lab = self.Labeler(proj)
        self.proj = proj 
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(Bit4DependencyParser, cls).add_arguments(argparser)
        argparser.add_argument('--proj', default=None, type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.BIT, tree.REL = self.lab.encode(tree)
            tree.transformed = True 
        return tree 

    def _pred(self, tree: CoNLL.Tree, bit_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(self.BIT.decode(bit_pred), self.REL.decode(rel_pred))
        return tree.rebuild_from_arcs(rec).deprojectivize(self.proj), well_formed        

    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        proj: Optional[str] = None,
        device: int = 0,
        **_
    ) -> Bit4DependencyParser:
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
            
        bit_tkz, rel_tkz = TargetTokenizer('BIT'), TargetTokenizer('REL')
        labeler = cls.Labeler(proj)
        bits, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bit_tkz.train(bits)
        rel_tkz.train(rels)
        
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        return cls(input_tkzs, [bit_tkz, rel_tkz], [enc_conf, *in_confs, bit_tkz.conf, rel_conf], proj, device)
