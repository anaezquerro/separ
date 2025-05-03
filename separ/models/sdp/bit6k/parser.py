from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Optional, Union, Tuple
import os 

from separ.data import SDP, EnhancedCoNLL, InputTokenizer, TargetTokenizer, Arc, adjacent_from_arcs, Graph
from separ.utils import Config, split, flatten
from separ.models.sdp.parser import SemanticSLParser


class Bit6kSemanticParser(SemanticSLParser):
    """6k-bit Semantic Parser from [Ezquerro et al., (2024)](https://aclanthology.org/2024.emnlp-main.659/)."""
    NAME = 'sdp-bit6k'
    PARAMS = ['k', 'join_rels', 'root_rel']
    DECOLLAPSE = True 
    
    class Labeler(SemanticSLParser.Labeler):
        """6k-bit encoding.
        - b0: word is a left dependant in a plane.
        - b1: word is a right dependant in a plane.
        - b2: word is the farthest left dependant in a plane.
        - b3: word is the farthest right dependant in a plane.
        - b4: word has left dependants in a plane.
        - b5: word has right dependants in a plane.
        """
        N_BITS = 6
        
        def __init__(self, k: int = 3):
            self.k = k 
            
        def __repr__(self) -> str:
            return f'Bit6kSemanticLabeler(k={self.k})'
        
        def recoverable(self, graph: Graph) -> List[Arc]:
            return flatten(self.preprocess(graph))
                 
        def preprocess(self, graph: Graph) -> List[List[Arc]]:
            """Divides the graph in a set of 6k-bit planes. Two arcs cannot belong to the same 
            plane if:
                1. They cross each other in the same direction.
                2. They share the same dependant in the same direction.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                List[List[Arc]]: List of k planes.
            """
            planes = [plane.copy() for plane in graph.bit6k_planes[:self.k]]
            if len(planes) < self.k:
                planes += [[] for _ in range(len(planes), self.k)]
            return planes
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            """Encodes a semantic graph with the 6k-bit representation.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                List[str]: Sequence of 6k-bit labels.
            """
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            planes = self.preprocess(graph)
            n = len(graph)
            labels = []
            for label in zip(*[self._encode(planes[p], n) for p in range(self.k)]):
                labels.append(''.join(label))
            return labels, self.encode_rels(flatten(planes), len(graph))
        
        def decode(self, labels: List[str], rels: Optional[List[str]] = None) -> Tuple[List[Arc], bool]:
            """Decodes the 6k-bit representation for a full sequence of labels.

            Args:
                labels (List[str]): 6k-bit input sequence.

            Returns:
                Tuple[List[Arc], torch.Tensor, bool]: Decoded arcs, adjacent matrix and whether the 
                    input sequence produces a well-formed semantic graph.
            """
            planes = zip(*[[''.join(lab) for lab in split(list(label), self.N_BITS)] for label in labels])
            arcs, well_formed = zip(*map(self._decode, planes))
            arcs = sorted(set(flatten(arcs)))
            if rels is not None:
                arcs = self.decode_rels(arcs, rels)
            return arcs, all(well_formed)
            
        def _encode(self, plane: List[Arc], n: int) -> List[str]:
            """Represents a plane with the 6-bit encoding.

            Args:
                plane (List[Arc]): List of non-crossing arcs.
                n (int): Number of nodes in the graph.
                
            Returns:
                List[str]: 6-bit representation.
            """
            labels = [[False for _ in range(self.N_BITS)] for _ in range(n)]
            adjacent = adjacent_from_arcs(plane, n)
            for arc in plane:
                if arc.DEP < arc.HEAD: # left arc 
                    # b0: DEP is a left dependant in the plane 
                    labels[arc.DEP-1][0] = True 
                    # b2: DEP is the farthest left dependant in the plane
                    labels[arc.DEP-1][2] = not adjacent[:arc.DEP, arc.HEAD].any()
                    # b4: HEAD has left dependants in the plane 
                    labels[arc.HEAD-1][4] = True 
                else: # right arc 
                    # b1: DEP is a right dependant in the plane 
                    labels[arc.DEP-1][1] = True 
                    # b3: DEP is the farthest right dependant in the plane 
                    labels[arc.DEP-1][3] = not adjacent[(arc.DEP+1):, arc.HEAD].any()
                    # b5: HEAD has right dependants in the plane 
                    if arc.HEAD != 0:
                        labels[arc.HEAD-1][5] = True 
            return [''.join(str(int(bit)) for bit in label) for label in labels]
                
        def _decode(self, labels: List[str]) -> Tuple[List[Arc], bool]:
            right, left, arcs = [0], [], []
            for idep, label in enumerate(labels):
                dep = idep+1
                label = map(bool, map(int, label))
                b0, b1, b2, b3, b4, b5 = label 
                if b1 and len(right) > 0: # DEP is a right dependant in the plane
                    arcs.append(Arc(right[-1], dep, None))
                    if b3 and right[-1] != 0: # DEP is the farthest right dependant in the plane
                        right.pop(-1)

                if b4: # DEP has left dependants in the plane 
                    last = False 
                    while not last and len(left) > 0:
                        last = left[-1][-1]
                        arcs.append(Arc(dep, left.pop(-1)[0], None))
                        
                if b5: # DEP has right dependants in the plane 
                    right.append(dep)
                    
                if b0: # DEP is a left dependant in the plane 
                    left.append((dep, b2))
            right.pop(0) # pop the node w0
            return arcs, len(right) == len(left) == 0
            
    def __init__(
        self,
        input_tkzs: List[InputTokenizer], 
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        join_rels: bool,
        root_rel: Optional[str],
        k: int,
        device: int 
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, join_rels, root_rel, device)
        self.k = k
        self.lab = self.Labeler(k)
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = SemanticSLParser.add_arguments(argparser)
        argparser.add_argument('-k', type=int, default=3, help='Number of planes for the 6-bit encoding')
        return argparser 
    
    @classmethod
    def build(
        cls,
        data: Union[SDP, EnhancedCoNLL, str], 
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        k: int = 2,
        join_rels: bool = False,
        device: int = 0,
        **_
    ) -> Bit6kSemanticParser:
        if isinstance(data, str):
            data = cls.load_data(data)
        
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        bit_tkz, rel_tkz, rel_conf, root_rel = cls.build_targets(
            label_tkz=TargetTokenizer('BIT'), rel_tkz=TargetTokenizer('REL'), 
            labeler=cls.Labeler(k=k), data=data, join_rels=join_rels
        )
        return cls(input_tkzs, [bit_tkz, rel_tkz], [enc_conf, *in_confs, bit_tkz.conf, rel_conf], join_rels, root_rel, k, device)