from __future__ import annotations
from argparse import ArgumentParser

from separ.data import Arc, adjacent_from_arcs, Graph
from separ.utils import split, flatten
from separ.models.grp.sl import GraphSLParser


class Bit6kGraphParser(GraphSLParser):
    """6k-bit Graph Parser from [Ezquerro et al., (2024)](https://aclanthology.org/2024.emnlp-main.659/)."""
    NAME = 'grp-bit6k'
    DECOLLAPSE = True 
    
    class Labeler(GraphSLParser.Labeler):
        """6k-bit encoding.
        - b0: word is a left dependant in a plane.
        - b1: word is a right dependant in a plane.
        - b2: word is the farthest left dependant in a plane.
        - b3: word is the farthest right dependant in a plane.
        - b4: word has left dependants in a plane.
        - b5: word has right dependants in a plane.
        """
        N_BITS = 6
        PARAMS = {'k'}

        def __init__(self, k: int = 3, **_):
            self.k = k 
            
        def __repr__(self) -> str:
            return f'Bit6kSemanticLabeler(k={self.k})'
        
        @property
        def NAMES(self) -> list[str]:
            return [f'6({self.k})BIT', 'REL']
        
        @property
        def NAME(self) -> str:
            return f'grp-6({self.k})bit'
        
        def recoverable(self, graph: Graph) -> list[Arc]:
            return flatten(self.preprocess(graph))
                 
        def preprocess(self, graph: Graph) -> list[list[Arc]]:
            """Divides the graph in a set of 6k-bit planes. Two arcs cannot belong to the same 
            plane if:
                1. They cross each other in the same direction.
                2. They share the same dependant in the same direction.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                list[list[Arc]]: List of k planes.
            """
            planes = [plane.copy() for plane in graph.bit6k_planes[:self.k]]
            if len(planes) < self.k:
                planes += [[] for _ in range(len(planes), self.k)]
            return planes
        
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            """Encodes a semantic graph with the 6k-bit representation.

            Args:
                graph (Graph): Input semantic graph.

            Returns:
                list[str]: Sequence of 6k-bit labels.
            """
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            planes = self.preprocess(graph)
            n = len(graph)
            labels = []
            for label in zip(*[self._encode(planes[p], n) for p in range(self.k)]):
                labels.append(''.join(label))
            return labels, self.encode_rels(flatten(planes), len(graph))
        
        def decode(self, labels: list[str], rels: list[str] | None = None) -> tuple[list[Arc], bool]:
            """Decodes the 6k-bit representation for a full sequence of labels.

            Args:
                labels (list[str]): 6k-bit input sequence.

            Returns:
                tuple[list[Arc], torch.Tensor, bool]: Decoded arcs, adjacent matrix and whether the 
                    input sequence produces a well-formed semantic graph.
            """
            planes = zip(*[[''.join(lab) for lab in split(list(label), self.N_BITS)] for label in labels])
            arcs, well_formed = zip(*map(self._decode, planes))
            arcs = sorted(set(flatten(arcs)))
            if rels is not None:
                arcs = self.decode_rels(arcs, rels)
            return arcs, all(well_formed)
            
        def _encode(self, plane: list[Arc], n: int) -> list[str]:
            """Represents a plane with the 6-bit encoding.

            Args:
                plane (list[Arc]): List of non-crossing arcs.
                n (int): Number of nodes in the graph.
                
            Returns:
                list[str]: 6-bit representation.
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
                
        def _decode(self, labels: list[str]) -> tuple[list[Arc], bool]:
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
            
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = GraphSLParser.add_arguments(argparser)
        argparser.add_argument('-k', type=int, default=3, help='Number of planes for the 6-bit encoding')
        return argparser 
    
