from __future__ import annotations
import numpy as np 
from typing import List, Set, Optional, Union, Dict

from trasepar.structs import adjacent_from_arcs, Graph, AbstractNode, Arc, AbstractDataset
from trasepar.utils import parallel

class SDP(AbstractDataset):
    """Representation of the SDP format (https://alt.qcri.org/semeval2015/task18/)"""
    EXTENSION = 'sdp'
    SEP = '\n\n'
    END = '\n\n'
    HEADER = '#SDP 2015'
    
    class Node(AbstractNode):
        FIELDS = ['ID', 'FORM', 'LEMMA', 'POS', 'TOP', 'PRED', 'FRAME', 'ARGS']
        
        def __init__(self, ID: int, FORM: str, LEMMA: str, POS: str, TOP: str, PRED: str, FRAME: str, *ARGS: List[str]):
            super().__init__(int(ID))
            self.FORM = FORM 
            self.LEMMA = LEMMA 
            self.POS = POS 
            self.TOP = TOP 
            self.PRED = PRED
            self.FRAME = FRAME
            self.ARGS = ARGS

        @property
        def is_root(self) -> bool:
            return self.TOP == '+'
        
        @property 
        def is_pred(self) -> bool:
            return self.PRED == '+'
        
        @property
        def n_heads(self) -> int:
            return sum(arg != '_' for arg in self.ARGS)
        
    class Graph(Graph):
        """Abstract representation of the Semantic Dependency Graph."""
        
        def __init__(self, nodes: List[SDP.Node], arcs: List[Arc], annotations: List[Union[int, str]] = [], ID: Optional[int] = None):
            self.NODE = SDP.Node
            super().__init__(nodes, arcs, annotations, ID)
            assert all(self.n_preds == len(node.ARGS) for node in self.nodes), f'{self.n_preds}\n' + '\n'.join(node.format() for node in self.nodes)
            assert self.n_preds == len(set(arc.HEAD for arc in self.arcs if arc.HEAD != 0)), f'{self.format()}\n{self.n_preds}\n{self.arcs}'
            assert self.n_roots == sum(node.is_root for node in self.nodes) == sum(arc.HEAD == 0 for arc in self.arcs)
        
        def rebuild(self, new_arcs: List[Arc]) -> SDP.Graph:
            """Rebuilds the graph from an input list of new arcs.

            Args:
                new_arcs (List[Arc]): New list of arcs.

            Returns:
                SDP.Graph: Resulting graph.
            """
            nodes = [node.copy() for node in self.nodes]
            for node in nodes:
                node.TOP = '-'
                node.PRED = '-'
            preds = sorted(set(arc.HEAD for arc in new_arcs if arc.HEAD != 0))
            preds = dict(zip(preds, range(len(preds))))
                
            for node in nodes:
                node.ARGS = ['_' for _ in range(len(preds))]
            for arc in new_arcs:
                if arc.HEAD == 0:
                    nodes[arc.DEP-1].TOP = '+'
                else:
                    nodes[arc.HEAD-1].PRED = '+'
                    nodes[arc.DEP-1].ARGS[preds[arc.HEAD]] = arc.REL
            return SDP.Graph(nodes, new_arcs, self.annotations, self.ID)
        

        @property 
        def bit4k_planes(self) -> List[List[Arc]]:
            if self._bit4k_planes is None:
                planes = [[]]
                for arc in sorted(self.arcs, key=lambda arc: (arc.left, arc.right)):
                    added = False 
                    for plane in planes:
                        if not any((arc.cross(other) and arc.side == other.side) 
                            or (arc.DEP == other.DEP) 
                            for other in plane):
                            plane.append(arc)
                            added = True 
                            break
                    if not added:
                        planes.append([arc])
                self._bit4k_planes = [planes[0]] + sorted(planes[1:], key=len, reverse=True)
            return self._bit4k_planes 
        
        @property 
        def bit6k_planes(self) -> List[List[Arc]]:
            if self._bit6k_planes is None:
                planes = [[]]
                for arc in sorted(self.arcs, key=lambda arc: (arc.left, arc.right)):
                    added = False 
                    for plane in planes:
                        if not any(
                            (arc.cross(other) and arc.side == other.side)        # arc crossing in the same direction
                            or (arc.DEP == other.DEP and arc.side == other.side) # arc with the same dependant and direction
                            for other in plane):
                            plane.append(arc)
                            added = True 
                            break 
                    if not added:
                        planes.append([arc])
                self._bit6k_planes = [planes[0]] + sorted(planes[1:], key=len, reverse=True)
            return self._bit6k_planes

        @property 
        def HEAD(self) -> List[List[int]]:
            """Returns the list of parents positions for each word.

            Returns:
                List[List[int]]: ``[n_words]``. List of parents positions for each word in the graph.
            """
            heads = [[] for _ in range(len(self))]
            for arc in self.arcs:
                heads[arc.DEP-1].append(arc.HEAD)
            return heads
        
        @property 
        def DEPS(self) -> List[List[int]]:
            deps = [[] for _ in range(len(self) + 1)]
            for arc in self.arcs:
                deps[arc.HEAD].append(arc.DEP)
            return deps
                
        @property
        def tags(self) -> List[str]:
            return [node.POS for node in self.nodes]
                    
        @classmethod
        def from_raw(cls, lines: str) -> SDP.Graph:
            annotations, nodes = [], [] 
            for line in lines.strip().split(cls.SEP):
                if line.split()[0].isdigit():
                    annotations.append(len(nodes))
                    nodes.append(SDP.Node(*line.split(SDP.Node.SEP)))
                elif line.startswith('#'):
                    annotations.append(line)
            # obtain those nodes that are predicates 
            preds = [node.ID for node in nodes if node.is_pred]
            
            # obtain root arcs 
            arcs = [Arc(0, node.ID, 'TOP') for node in nodes if node.is_root]
            arcs += [Arc(pred, node.ID, arg) for node in nodes for arg, pred in zip(node.ARGS, preds) if arg != '_']
            return cls(nodes, sorted(arcs), annotations)
        
        @property 
        def n_preds(self) -> int:
            return sum(node.is_pred for node in self.nodes)
        
        @property 
        def n_roots(self) -> int:
            return sum(arc.HEAD == 0 for arc in self.arcs)


    @classmethod 
    def from_file(cls, path: str, num_workers: int = 1) -> SDP:
        data = '\n'.join(open(path, 'r').read().split('\n')[1:]).strip() # get data and remove header
        blocks = [block for block in data.split('\n\n') if '\n' in block]
        graphs = parallel(cls.Graph.from_raw, blocks, num_workers=num_workers, name=path.split('/')[-1])
        return cls(graphs, path)
    
    @property
    def n_arcs(self) -> int:
        return sum(len(graph.arcs) for graph in self)