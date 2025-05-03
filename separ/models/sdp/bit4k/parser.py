from __future__ import annotations
import torch
from typing import List, Tuple

from separ.data import Arc, adjacent_from_arcs, Graph
from separ.models.sdp.bit6k.parser import Bit6kSemanticParser
from separ.utils import flatten, split 


class Bit4kSemanticParser(Bit6kSemanticParser):
    """4k-bit Semantic Parser from [Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)."""
    NAME = 'sdp-bit4k'
    DECOLLAPSE = True 
    
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
        
        def recoverable(self, graph: Graph) -> List[Arc]:
            return flatten(graph.bit4k_planes[p] for p in range(min(self.k, len(graph.bit4k_planes))))
        
        def fill_artificial(self, arcs: List[Arc], n: int) -> List[Arc]:
            """Fills a list of arcs with artificial nodes to satisfy that each node 
            has one and only one parent.

            Args:
                arcs (List[Arc]): List of arcs.
                n (int): Graph size.

            Returns:
                List[Arc]: New list of arcs.
            """
            no_assigned = set(range(1, n+1)) - set(arc.DEP for arc in arcs)
            for dep in no_assigned:
                arcs.append(Arc(dep-1, dep, self.EMPTY_REL))
            return sorted(arcs)
        
        def preprocess(self, graph: Graph) -> List[List[Arc]]:
            """Divides the graph in a set of 4k-bit planes. Two arcs cannot belong to the same plane
            if:
                1. The cross each other in the same direction.
                2. They share the same dependant.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                List[List[Arc]]: List of k planes.
            """
            planes = [plane.copy() for plane in graph.bit4k_planes[:self.k]]
            if len(planes) < self.k:
                planes += [[] for _ in range(len(planes), self.k)]
                
            # each node must have a head in each plane (assign by default the previous one)
            planes = [self.fill_artificial(plane, len(graph)) for plane in planes]
            return planes
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            """Encodes a semantic graph with the 4k-bit representation.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                List[str]: Sequence of 4k-bit labels.
            """
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            planes = self.preprocess(graph)
            n = len(graph)
            labels = []
            for label in zip(*[self._encode(planes[p], n) for p in range(self.k)]):
                labels.append(''.join(label))
            # encode arc relations, skip repeated arcs or those with a null label in some plane
            filtered = [arc for arc in set(flatten(planes)) if arc.REL != self.EMPTY_REL or not graph.ADJACENT[arc.DEP, arc.HEAD]]
            return labels, self.encode_rels(filtered, len(graph))
        
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
        
        def decode_rels(self, arcs: List[Arc], rels: List[str]) -> List[Arc]:
            rels = [rel.split(self.SEP) for rel in rels]
            filtered = []
            for arc in arcs:
                arc.REL = rels[arc.DEP-1].pop(0) if len(rels[arc.DEP-1]) > 0 else self.REL 
                if arc.REL != self.EMPTY_REL:
                    filtered.append(arc)
            return filtered
                
        def _decode(self, labels: List[str]) -> Tuple[List[Arc], bool]:
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
        
    
    def transform(self, graph: Graph) -> Graph:
        if not graph.transformed:
            labels, rels = self.lab.encode(graph)
            graph.__setattr__(self.LABEL.name, labels)
            if self.join_rels:
                graph.REL = rels
            else:
                arcs = [arc for arc in set(flatten(self.lab.preprocess(graph)))\
                    if arc.REL != self.lab.EMPTY_REL or not graph.ADJACENT[arc.DEP, arc.HEAD]]
                graph.REL = [arc.REL for arc in arcs]
                graph.MATRIX = adjacent_from_arcs(arcs, len(graph))
            graph.transformed = True 
        return graph 
    
    def _pred_rel(self, graph: Graph, arc_pred: List[Arc], rel_pred: torch.Tensor) -> Graph:
        rel_pred = self.REL.decode(rel_pred)
        arc_filtered = []
        for arc in arc_pred:
            arc.REL = rel_pred.pop(0) 
            if arc.HEAD == 0 and self.root_rel:
                arc.REL = self.root_rel 
            if arc.REL != self.lab.EMPTY_REL:
                arc_filtered.append(arc)
        return graph.rebuild_from_arcs(Graph.decollapse_one_cycles(arc_filtered))