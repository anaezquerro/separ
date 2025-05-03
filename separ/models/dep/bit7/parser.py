from __future__ import annotations
from typing import Tuple, Set, List, Set, Optional, Union
import torch 

from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc
from separ.utils import flatten, Config, bar 
from separ.models.dep.bit4.parser import Bit4DependencyParser
from separ.models.dep.parser import DependencySLParser


class Bit7DependencyParser(Bit4DependencyParser):
    """7-bit Dependency Parser from [Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)."""
    NAME = 'dep-bit7'

    class Labeler(DependencySLParser.Labeler):
        """7-bit encoding for 2-planar dependency trees.
        
        b0: word is a left dependant (0) or a right dependant (1).
        b1: word is a dependant in the first (0) or second (1) plane.
        b2: word is the outermost dependant of its parent in its plane.
        b3: word has left dependants in the first plane.
        b4: word has right dependants in the first plane.
        b5: word has left dependants in the second plane.
        b6: word has right dependants in the second plane.
        """
        NUM_BITS = 7
        DEFAULT = '0000000'
        
        def __init__(self):
            pass 
        
        def __repr__(self):
            return f'Bit7DependencyLabeler()'
        
        def preprocess(self, graph: CoNLL.Graph) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
            """Obtain the dependants of each plane.

            Args:
                graph (CoNLL.Graph): Input graph.

            Returns:
                Tuple[List[int], List[int], List[int], List[int]]:: Dependants and heads of plane 1 and plane 2.
            """
            if len(graph.planes) < 2:
                return set(arc.DEP for arc in graph.planes[0]), set(arc.HEAD for arc in graph.planes[0]), set(), set()
            elif len(graph.planes) == 2:
                plane1, plane2 = graph.planes[0], graph.planes[1]
            else:
                plane1 = [arc for p in range(len(graph.planes)) for arc in graph.planes[p] if p != 1]
                plane2 = graph.planes[1]
            return set(arc.DEP for arc in plane1), set(arc.HEAD for arc in plane1),\
                set(arc.DEP for arc in plane2), set(arc.HEAD for arc in plane2)
                
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
            deps1, heads1, deps2, heads2 = self.preprocess(graph)
            labels = [[False for _ in range(self.NUM_BITS)] for _ in range(len(graph))]
            for idep, head in enumerate(graph.HEAD):
                dep = idep + 1
                labels[idep][0] = head < dep 
                labels[idep][1] = dep in deps2 
                if dep in heads1:
                    labels[idep][3] = len(set(graph.ADJACENT[:dep, dep].nonzero().unique().tolist()) & deps1) > 0
                    labels[idep][4] = len(set((dep + graph.ADJACENT[dep:, dep].nonzero().unique()).tolist()) & deps1) > 0
                if dep in heads2:
                    labels[idep][5] = len(set(graph.ADJACENT[:dep, dep].nonzero().unique().tolist()) & deps2) > 0
                    labels[idep][6] = len(set((dep + graph.ADJACENT[dep:, dep].nonzero().unique()).tolist()) & deps2) > 0
                if dep in deps1:
                    others = set(graph.ADJACENT[:, head].nonzero().unique().tolist()) & deps1
                else:
                    others = set(graph.ADJACENT[:, head].nonzero().unique().tolist()) & deps2 
                labels[idep][2] = (max(others) if head < dep else min(others)) == dep
            return [''.join(map(str, map(int, label))) for label in labels], list(graph.DEPREL)
        
        def decode(self, bits: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            left1, right1 = [], [0]
            left2, right2 = [], []
            well_formed = True 
            adjacent = torch.zeros(len(bits)+1, len(bits)+1, dtype=torch.bool)
            for idep, label in enumerate(bits):
                b0, b1, b2, b3, b4, b5, b6 = map(bool, map(int, label))
                
                # right dependant in the first plane 
                if b0 and not b1:
                    if len(right1) > 0 and self.is_valid(adjacent, idep+1, right1[-1]):
                        adjacent[idep+1, right1[-1]] = True 
                        if b2: # farthest dependant 
                            right1.pop(-1)
                    else:
                        well_formed = False
                            
                # right dependant in the second plane 
                if b0 and b1:
                    if len(right2) > 0 and self.is_valid(adjacent, idep+1, right2[-1]):
                        adjacent[idep+1, right2[-1]] = True 
                        if b2: # farthest dependant 
                            right2.pop(-1)
                    else:
                        well_formed = False
                            
                 # word has left dependants in the first plane 
                if b3:
                    last = False 
                    while not last: 
                        if len(left1) > 0 and self.is_valid(adjacent, left1[-1][0], idep+1):
                            dep, last = left1.pop(-1)
                            adjacent[dep, idep+1] = True
                        else:
                            well_formed = False 
                            break 
                    
                # word has left dependants in the second plane 
                if b5:
                    last = False
                    while not last:
                        if len(left2) > 0 and self.is_valid(adjacent, left2[-1][0], idep+1):
                            dep, last = left2.pop(-1)
                            adjacent[dep, idep+1] = True
                        else:
                            well_formed = False 
                            break
                
                # left dependant in the first plane 
                if not b0 and not b1: 
                    left1.append((idep+1, b2))
                    
                # left dependant in the second plane 
                if not b0 and b1: 
                    left2.append((idep+1, b2))
                    
                # word has right dependants in the first plane 
                if b4:
                    right1.append(idep+1)
                
                # word has right dependants in the second plane 
                if b6: 
                    right2.append(idep+1)
            return self.postprocess(adjacent, rels), well_formed and (len(right1 + left1 + right2 + left2) == 0) and (adjacent.sum() == len(bits))
        
        def test(self, graph: CoNLL.Graph) -> bool:
            return super().test(graph.planarize(2))
            
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        device: int
    ):
        super(Bit4DependencyParser, self).__init__(input_tkzs, target_tkzs, model_confs, device)
        self.lab = self.Labeler()
        
    def _pred(self, tree: CoNLL.Tree, bit_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(self.BIT.decode(bit_pred), self.REL.decode(rel_pred))
        return tree.rebuild_from_arcs(rec), well_formed        
    
    @classmethod
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: int = 0,
        **_
    ) -> Bit7DependencyParser:
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
        labeler = cls.Labeler()
        bits, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bit_tkz.train(bits)
        rel_tkz.train(rels)
        
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        return cls(input_tkzs, [bit_tkz, rel_tkz], [enc_conf, *in_confs, bit_tkz.conf, rel_conf], device)