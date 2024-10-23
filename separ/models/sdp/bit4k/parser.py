from __future__ import annotations
import torch
from typing import List, Tuple, Set

from separ.data import SDP, Tokenizer
from separ.structs import Arc, adjacent_from_arcs
from separ.utils import flatten, SemanticMetric
from separ.models.sdp.bit4k.model import Bit4kSemanticModel
from separ.models.sdp.bit6k.parser import Bit6kSemanticParser


class Bit4kSemanticParser(Bit6kSemanticParser):
    """4k-bit Semantic Parser from [Ezquerro et al., 2024]()."""
    NAME = 'sdp-bit4k'
    DATASET = SDP 
    METRIC = SemanticMetric
    MODEL = Bit4kSemanticModel
    PARAMS = ['k']
    
    class Labeler(Bit6kSemanticParser.Labeler):
        """4k-bit encoding.
        - b0: word is a left dependant (0) or a right dependant (1) in the plane.
        - b1: word is the farthest dependant in the plane.
        - b2: word has left dependants in the plane.
        - b3: word has right dependants in the plane.
        """
        EMPTY_REL = 'NULL'
        N_BITS = 4
        
        def __repr__(self) -> str:
            return f'Bit4kSemanticLabeler(k={self.k})'
            
        @property 
        def DEFAULT(self) -> str:
            return '0000'*self.k
        
        def preprocess(self, graph: SDP.Graph) -> List[List[Arc]]:
            """Divides the graph in a set of 4k-bit planes. Two arcs cannot belong to the same plane
            if:
                1. The cross each other in the same direction.
                2. They share the same dependant.

            Args:
                graph (SDP.Graph): Input semantic graph.

            Returns:
                List[List[Arc]]: List of k planes.
            """
            if not graph.bit4k_planes:
                planes = [[]]
                adjacent = [adjacent_from_arcs([], len(graph))]
                for arc in sorted(graph.arcs, key=lambda arc: arc.left):
                    added = False
                    for p in range(len(planes)):
                        # the arc cannot cross any other arc in the same direction 
                        # there cannot be other arc with the same dependant 
                        if not any(arc.cross(other) and arc.side == other.side for other in planes[p]) and \
                            not adjacent[p][arc.DEP].any():
                            planes[p].append(arc)
                            adjacent[p][arc.DEP, arc.HEAD] = True 
                            added = True 
                            break 
                    if not added:
                        planes.append([arc])
                        adjacent.append(adjacent_from_arcs(planes[-1], len(graph)))
                        
                if len(planes) < self.k: # add empty planes 
                    planes += [[] for _ in range(len(planes), self.k)]
                elif len(planes) > self.k:
                    # sort planes by the number of arcs and index
                    order = sorted(range(len(planes)), key=lambda i: (len(planes[i]), -i), reverse=True)
                    planes = [planes[p] for p in order[:self.k]]
                    
                # each node must have a head in each plane (assign by default the previous one)
                for plane in planes:
                    deps = set(range(1, len(graph)+1)) - set(arc.DEP for arc in plane)
                    plane += [Arc(dep-1 if dep != 1 or len(graph) == 1 else 2, dep, self.EMPTY_REL) for dep in deps]
                graph.bit4k_planes = planes 
            return graph.bit4k_planes
        
        def _encode(self, plane: List[Arc], n: int) -> List[str]:
            labels = [[False for _ in range(self.N_BITS)] for _ in range(n)]
            adjacent = adjacent_from_arcs(plane, n)
            for arc in plane:
                if arc.DEP < arc.HEAD: # left arc 
                    # b2: arc.HEAD has left dependants in the plane
                    labels[arc.HEAD-1][2] = True 
                if arc.HEAD < arc.DEP: # right arc 
                    labels[arc.DEP-1][0] = True # b0: arc.DEP is a right dependant 
                    # b3: arc.HEAD has right dependants in the plane
                    if arc.HEAD != 0:
                        labels[arc.HEAD-1][3] = True 
                # b1: arc.DEP is the farthest dependant in the plane
                if not (adjacent[arc.DEP+1:, arc.HEAD] if arc.side == 1 else adjacent[:arc.DEP, arc.HEAD]).any(): 
                    labels[arc.DEP-1][1] = True 
            return [''.join(str(int(bit)) for bit in label) for label in labels]
        
        def _decode(self, labels: List[str])-> List[Arc]:
            right, left, arcs = [0], [], []
            for idep, label in enumerate(labels):
                dep = idep+1
                b0, b1, b2, b3 = map(bool, map(int, label)) 
                
                if b0: # DEP is a right dependant in the plane 
                    arcs.append(Arc(right[-1], dep, None))
                    if b1 and right[-1] != 0: # DEP is the farthest dependant
                        right.pop(-1)
                
                if b2: # DEP has left dependants in the plane 
                    last = False 
                    while not last:
                        last = left[-1][-1]
                        arcs.append(Arc(dep, left.pop(-1)[0], None))
                if b3: # DEP has right dependants in the plane
                    right.append(dep)
                if not b0: # DEP is a left dependant in the plane 
                    left.append((dep, b1))
            return arcs
                
        def _decode_postprocess(self, labels: List[str]) -> Tuple[List[Arc], bool]:
            right, left, arcs = [0], [], []
            for idep, label in enumerate(labels):
                dep = idep+1
                b0, b1, b2, b3 = map(bool, map(int, label)) 
                
                if b0 and len(right) > 0: # DEP is a right dependant in the plane 
                    arcs.append(Arc(right[-1], dep, None))
                    if b1 and right[-1] != 0: # DEP is the farthest dependant
                        right.pop(-1)
                
                if b2: # DEP has left dependants in the plane 
                    last = False 
                    while not last and len(left) > 0:
                        last = left[-1][-1]
                        arcs.append(Arc(dep, left.pop(-1)[0], None))
                if b3: # DEP has right dependants in the plane
                    right.append(dep)
                if not b0: # DEP is a left dependant in the plane 
                    left.append((dep, b1))
            right.pop(0)
            return arcs, len(right) == len(left) == 0
        
        def theoretical(self, graph: SDP.Graph) -> SDP.Graph:
            planes = self.preprocess(graph)
            arcs = [arc for arc in flatten(planes) if arc.REL != self.EMPTY_REL]
            new = graph.rebuild(arcs)
            return new
        
        def empirical(self, graph: SDP.Graph, labels: Set[str], REL: str) -> SDP.Graph:
            bits = [bit if bit in labels else self.DEFAULT for bit in self.encode(graph)]
            arcs = sorted(flatten(*self.preprocess(graph)))
            expanded = graph.rebuild(arcs)
            recovered = []
            for arc in self.decode_postprocess(bits)[0]:
                if graph.ADJACENT[arc.DEP, arc.HEAD]:
                    arc.REL = graph.LABELED_ADJACENT[arc.DEP, arc.REL]
                elif not expanded.ADJACENT[arc.DEP, arc.HEAD]:
                    arc.REL = REL 
            return graph.rebuild(recovered)
        
        def complete_decode_postprocess(self, labels: List[str], rels: List[str]) -> Tuple[List[Arc], bool]: 
            expanded, _, well_formed = self.decode_postprocess(labels)
            rels = [rel.split(self.SEP) for rel in rels]
            recovered = []
            for arc in expanded: # arcs are already sorted
                if len(rels[arc.DEP-1]) > 0:
                    arc.REL = rels[arc.DEP-1].pop(0)
                else:
                    arc.REL = self.DEFAULT_REL
                    well_formed = False
                if arc.REL != self.EMPTY_REL:
                    recovered.append(arc)
            return recovered, well_formed
    
    @classmethod
    def transform(cls, graph: SDP.Graph, input_tkzs: List[Tokenizer], BIT: Tokenizer, REL: Tokenizer, labeler: Bit4kSemanticParser.Labeler):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            # obtain the extra arcs with NULL dependency labels
            arcs = flatten(labeler.preprocess(graph))
            real = [arc for arc in arcs if arc.REL != labeler.EMPTY_REL]
            adjacent = adjacent_from_arcs(real, len(graph))
            for arc in arcs:
                if arc.REL == labeler.EMPTY_REL and not adjacent[arc.DEP, arc.HEAD]:
                    real.append(arc)
                    adjacent[arc.DEP, arc.HEAD] = True
            expanded = graph.rebuild(sorted(real))
            
            graph.BIT = BIT.encode(labeler.encode(graph)).pin_memory()
            graph.REL = REL.encode([arc.REL for arc in expanded.arcs if arc.HEAD != 0]).pin_memory()
            graph.MATRIX = expanded.ADJACENT[1:, 1:].pin_memory()
            graph._transformed = True 
    
    def _pred(self, graph: SDP.Graph, arc_pred: List[Arc], rel_pred: torch.Tensor) -> SDP.Graph:
        rel_pred = self.REL.decode(rel_pred)
        for arc in sorted(arc_pred):
            if arc.HEAD == 0:
                arc.REL = 'root'
            else:
                arc.REL = rel_pred.pop(0)
        arc_pred = [arc for arc in arc_pred if arc.REL != self.labeler.EMPTY_REL]
        return graph.rebuild(arc_pred)