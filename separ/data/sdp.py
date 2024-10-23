from __future__ import annotations
import numpy as np 
from typing import List, Set, Optional, Union, Dict

from separ.structs import Graph, AbstractNode, Arc, AbstractDataset
from separ.utils import parallel

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
            self.bit6k_planes: List[List[Arc]] = None 
            self.bit4k_planes: List[List[Arc]] = None 
            # assert all(self.n_preds == len(node.ARGS) for node in self.nodes), f'{self.n_preds}\n' + '\n'.join(node.format() for node in self.nodes)
            # assert self.n_preds == len(set(arc.HEAD for arc in self.arcs if arc.HEAD != 0)), f'{self.format()}\n{self.n_preds}\n{self.arcs}'
            # assert self.n_roots == sum(node.is_root for node in self.nodes) == sum(arc.HEAD == 0 for arc in self.arcs)
        
        def rebuild(self, new_arcs: List[Arc]) -> SDP.Graph:
            # new_arcs = sorted(new_arcs)
            
            # remove arcs that exceed the length of the graph and have repeated nodes 
            # GRAPH = np.zeros((len(self), len(self)+1), bool)
            # valid_arcs = []
            # for arc in new_arcs:
            #     if arc.DEP not in range(1, len(self) + 1):
            #         continue 
            #     if arc.HEAD not in range(len(self) + 1):
            #         continue 
            #     if GRAPH[arc.DEP-1, arc.HEAD]:
            #         continue 
            #     GRAPH[arc.DEP-1, arc.HEAD] = True
            #     valid_arcs.append(arc)
                
            nodes = [node.copy() for node in self.nodes]
            for node in nodes:
                node.TOP = '-'
                node.PRED = '-'
                
            preds = sorted(set(arc.HEAD for arc in new_arcs if arc.HEAD != 0))
            preds = dict(zip(preds, range(len(preds))))
                
            # ARGS = [['_' for _ in range(len(preds))] for _ in nodes]
            for node in nodes:
                node.ARGS = ['_' for _ in range(len(preds))]
            for arc in new_arcs:
                if arc.HEAD == 0:
                    nodes[arc.DEP-1].TOP = '+'
                else:
                    nodes[arc.HEAD-1].PRED = '+'
                    nodes[arc.DEP-1].ARGS[preds[arc.HEAD]] = arc.REL
            return SDP.Graph(nodes, new_arcs, self.annotations, self.ID)
        
        def rebuild_tags(self, tags: List[str]) -> SDP.Graph:
            nodes = []
            for node, tag in zip(self.nodes, tags):
                new = node.copy()
                new.POS = tag 
                nodes.append(new)
            return SDP.Graph(nodes, self.arcs, self.annotations, self.ID)

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
        def rels(self) -> Set[str]:
            return set(arc.REL for arc in self.arcs)
        
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