from __future__ import annotations

from separ.data.struct import Graph, Arc, Dataset
from separ.utils.logger import bar

class SDP(Dataset):
    """Representation of the SDP format (https://alt.qcri.org/semeval2015/task18/)"""
    EXTENSION = 'sdp'
    SEP = '\n\n'
    
    class Graph(Graph):
        """Abstract representation of the Semantic Dependency Graph."""
        class Node(Graph.Node):
            FIELDS = ['ID', 'FORM', 'LEMMA', 'POS', 'TOP', 'PRED', 'FRAME', 'ARGS']
            
            def __init__(self, ID: int, FORM: str, LEMMA: str, POS: str, TOP: str, PRED: str, FRAME: str, *ARGS: list[str]):
                self.ID = int(ID)
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
            
            def copy(self) -> Graph.Node:
                return self.__class__(self.ID, self.FORM, self.LEMMA, self.POS, self.TOP, self.PRED, self.FRAME, *self.ARGS)
            
            def __eq__(self, other: SDP.Graph.Node) -> bool:
                return self.format() == other.format()
        
        def __init__(
            self, 
            nodes: list[SDP.Graph.Node], 
            arcs: list[Arc], 
            ID: int | None = None,
            annotations: list[int | str] | None = None 
        ):
            super().__init__(nodes, arcs, ID, annotations)
            assert all(self.n_preds == len(node.ARGS) for node in self.nodes), f'{self.n_preds}\n' + '\n'.join(node.format() for node in self.nodes)
            assert self.n_preds == len(set(arc.HEAD for arc in self.arcs if arc.HEAD != 0)), f'{self.format()}\n{self.n_preds}\n{self.arcs}'
            assert self.n_roots == sum(node.is_root for node in self.nodes) == sum(arc.HEAD == 0 for arc in self.arcs)
        
        def rebuild_from_arcs(self, new_arcs: list[Arc]) -> SDP.Graph:
            """Rebuilds the graph from an input list of new arcs.

            Args:
                new_arcs (list[Arc]): New list of arcs.

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
            return SDP.Graph(nodes, new_arcs, ID=self.ID, annotations=self.annotations)
                
        @property
        def tags(self) -> list[str]:
            return [node.POS for node in self.nodes]
                    
        @classmethod
        def from_raw(cls, lines: str, ID: int) -> SDP.Graph:
            annotations, nodes = [], [] 
            for line in lines.strip().split(cls.SEP):
                if line.split()[0].isdigit():
                    annotations.append(len(nodes))
                    nodes.append(cls.Node(*line.split(cls.Node.SEP)))
                elif line.startswith('#'):
                    annotations.append(line)
            # obtain those nodes that are predicates 
            preds = [node.ID for node in nodes if node.is_pred]
            
            # obtain root arcs 
            arcs = [Arc(0, node.ID, 'TOP') for node in nodes if node.is_root]
            arcs += [Arc(pred, node.ID, arg) for node in nodes for arg, pred in zip(node.ARGS, preds) if arg != '_']
            return cls(nodes, sorted(arcs), ID=ID, annotations=annotations)
        
        @property 
        def n_preds(self) -> int:
            return sum(node.is_pred for node in self.nodes)
        

    @classmethod 
    def from_file(cls, path: str) -> SDP:
        data = '\n'.join(open(path, 'r').read().split('\n')[1:]).strip() # get data and remove header
        blocks = [block for block in data.split('\n\n') if '\n' in block]
        graphs = list(bar(map(cls.Graph.from_raw, blocks, range(len(blocks))), leave=False, total=len(blocks), desc=path))
        return cls(graphs, path)
    
    @property
    def n_arcs(self) -> int:
        """Number of arcs of the dataset."""
        return sum(len(graph.arcs) for graph in self)