from __future__ import annotations
from typing import Tuple, Set, List, Set
import torch 

from trasepar.data import CoNLL
from trasepar.structs import Arc, forms_cycles, candidates_no_cycles
from trasepar.models.dep.bit4.parser import Bit4DependencyParser
from trasepar.models.dep.bit7.model import Bit7DependencyModel
from trasepar.models.dep.labeler import DependencyLabeler


class Bit7DependencyParser(Bit4DependencyParser):
    """7-bit Dependency Parser from [Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)."""
    NAME = 'dep-bit7'
    MODEL = Bit7DependencyModel

    class Labeler(DependencyLabeler):
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
        
        def decode(self, bits: List[str], rels: List[str]) -> List[Arc]:
            left1, right1 = [], [0]
            left2, right2 = [], []
            arcs = []
            for idep, (label, rel) in enumerate(zip(bits, rels)):
                b0, b1, b2, b3, b4, b5, b6 = map(bool, map(int, label))
                if b0 and not b1: # right dependant in the first plane 
                    arcs.append(Arc(right1[-1], idep+1, rel))
                    if b2: # farthest dependant 
                        right1.pop(-1)
                if b0 and b1: # right dependant in the second plane 
                    arcs.append(Arc(right2[-1], idep+1, rel))
                    if b2: # farthest dependant 
                        right2.pop(-1)
                if b3: # word has left dependants in the first plane 
                    last = False
                    while not last:
                        last = left1[-1][-1]
                        arcs.append(Arc(idep+1, *left1.pop(-1)[:-1]))
                if b5: # word has left dependants in the second plane
                    last = False  
                    while not last:
                        last = left2[-1][-1]
                        arcs.append(Arc(idep+1, *left2.pop(-1)[:-1]))
                if not b0 and not b1: # left dependant in the first plane 
                    left1.append((idep+1, rel, b2))
                if not b0 and b1: # left dependant in the second plane 
                    left2.append((idep+1, rel, b2))
                if b4: # word has right dependants in the first plane 
                    right1.append(idep+1)
                if b6: # word has right dependants in the second plane 
                    right2.append(idep+1)
            return sorted(arcs)
        
        def decode_postprocess(self, bits: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            left1, right1 = [], [0]
            left2, right2 = [], []
            arcs = []
            well_formed = True 
            adjacent = torch.zeros(len(bits)+1, len(bits)+1, dtype=torch.bool)
            for idep, label in enumerate(bits):
                b0, b1, b2, b3, b4, b5, b6 = map(bool, map(int, label))
                
                # right dependant in the first plane 
                if b0 and not b1:
                    if len(right1) > 0 and not forms_cycles(adjacent, idep+1, right1[-1]) and (not adjacent[:, 0].any() or right1[-1] != 0):
                        arcs.append(Arc(right1[-1], idep+1, None))
                        adjacent[idep+1, right1[-1]] = True 
                        if b2: # farthest dependant 
                            right1.pop(-1)
                    else:
                        well_formed = False
                            
                # right dependant in the second plane 
                if b0 and b1:
                    if len(right2) > 0 and not forms_cycles(adjacent, idep+1, right2[-1]) and (not adjacent[:, 0].any() or right2[-1] != 0):
                        arcs.append(Arc(right2[-1], idep+1, None))
                        adjacent[idep+1, right2[-1]] = True 
                        if b2: # farthest dependant 
                            right2.pop(-1)
                    else:
                        well_formed = False
                            
                 # word has left dependants in the first plane 
                if b3:
                    last = False 
                    while not last: 
                        if len(left1) > 0 and not forms_cycles(adjacent, left1[-1][0], idep+1):
                            dep, last = left1.pop(-1)
                            arcs.append(Arc(idep+1, dep, None))
                            adjacent[dep, idep+1] = True
                        else:
                            well_formed = False 
                            break 
                    
                # word has left dependants in the second plane 
                if b5:
                    last = False
                    while not last:
                        if len(left2) > 0 and not forms_cycles(adjacent, left2[-1][0], idep+1):
                            dep, last = left2.pop(-1)
                            arcs.append(Arc(idep+1, dep, None))
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
            well_formed = well_formed and (len(right1 + left1 + right2 + left2) == 0) and len(arcs) == len(bits)
            arcs = sorted(self.postprocess(arcs, adjacent))
            for arc, rel in zip(arcs, rels):
                arc.REL = rel if arc.HEAD != 0 else 'root'
            return arcs, well_formed
        
        def postprocess(self, arcs: List[Arc], adjacent: torch.Tensor) -> List[Arc]:
            no_assigned = sorted(set((adjacent.sum(-1) == 0).nonzero().flatten().tolist()[1:]))
            for dep in no_assigned:
                head = candidates_no_cycles(adjacent, dep).pop(0)
                arcs.append(Arc(head, dep, None))
                adjacent[dep, head] = True
            return arcs 
        
        def test(self, graph: CoNLL.Graph) -> bool:
            if len(graph.planes) <= 2:
                bits, rels = self.encode(graph)
                rec1 = graph.rebuild(self.decode(bits, rels))
                rec2, well_formed = self.decode_postprocess(bits, rels)
                rec2 = graph.rebuild(rec2)
                return graph == rec1 == rec2 and well_formed
            else:
                planar = graph.planarize(2)
                bits, rels = self.encode(planar)
                rec1 = planar.rebuild(self.decode(bits, rels))
                rec2, well_formed = self.decode_postprocess(bits, rels)
                rec2 = planar.rebuild(rec2)
                return planar == rec1 == rec2 and well_formed

    
    