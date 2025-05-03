from __future__ import annotations
from typing import List, Optional, Union
import re 
import numpy as np 

from separ.utils.logger import bar
from separ.data.conll import CoNLL 
from separ.data.struct import Arc, Graph 
        
class EnhancedCoNLL(CoNLL):
    """Representation of the [Enhanced CoNLL-U](https://universaldependencies.org/v2/conll-u.html) format for Graph Parsing."""
    
    class Graph(Graph):
        CYCLE = '<c>'
        
        class Node(CoNLL.Tree.Node):
            ...
        
        def __init__(
            self, 
            nodes: List[EnhancedCoNLL.Graph.Node], 
            arcs: List[Arc], 
            ID: Optional[int] = None, 
            annotations: Optional[List[Union[str, int]]] = None
        ):
            super().__init__(nodes, arcs, ID, annotations)

        @property
        def POS(self) -> List[str]:
            return [node.UPOS for node in self.nodes]
        
        @property
        def tags(self) -> List[str]:
            return [node.UPOS for node in self.nodes]
        
        @classmethod
        def from_raw(cls, lines: str, ID: int) -> EnhancedCoNLL.Graph:
            """Parses a CoNLL block and extracts the enhanced dependencies. Repeated arcs (with the same head and 
            dependent) are collapsed in a single arc using the separator |. For instance, given two arcs in the same 
            graph: 3 -(r1)-> 1 and 3 -(r2)-> 1 are collapsed as 3 -(r1|r2)-> 1. In the DEPS field are separated.

            Args:
                lines (str): CoNLL block.

            Returns:
                EnhancedCoNLL.Graph.
            """
            lines = list(filter(lambda x: len(x) > 0, lines.strip().split('\n')))
            nodes, annotations = [], []
            for line in lines:
                if line.split()[0].isdigit():
                    annotations.append(len(nodes))
                    nodes.append(cls.Node.from_raw(line))
                elif line.startswith('#'):
                    annotations.append(line)
            
            # first obtain all arcs (even repeated ones)
            arcs = []
            for node in nodes:
                if node.DEPS == '_':
                    continue 
                values = node.DEPS.split('|')
                for value in values:
                    head, *rel = value.split(':')
                    rel = ':'.join(rel)
                    if not head.isdigit():
                        continue 
                    arcs.append(Arc(int(head), node.ID, rel))
            
            # remove repeated 
            arcs = sorted(set(arcs))
            adjacent = np.repeat('', (len(nodes)+1)**2).astype(np.object_).reshape(len(nodes)+1, len(nodes)+1)
            for arc in arcs:
                if adjacent[arc.DEP, arc.HEAD] == '':
                    adjacent[arc.DEP, arc.HEAD] = arc.REL
                else: # repeated arc
                    adjacent[arc.DEP, arc.HEAD] += f'|{arc.REL}'  
            
            # create deps
            arcs = []
            for node in nodes:
                node.DEPS = '|'.join(f'{head}:{r}' for head, rel in enumerate(adjacent[node.ID]) for r in rel.split('|') if len(rel) > 0)
                arcs += [Arc(head, node.ID, rel) for head, rel in enumerate(adjacent[node.ID]) if len(rel) > 0]
            return cls(nodes, arcs, ID=ID, annotations=annotations)
        
        def rebuild_from_arcs(self, arcs: List[Arc]) -> EnhancedCoNLL.Graph:
            """Rebuilds a graph with a new sequence of arcs, taking into account that repeated arcs must be 
            represented in the DEPS field.

            Args:
                arcs (List[Arc]): New list of arcs.

            Returns:
                EnhancedCoNLL.Graph: New graph.
            """
            nodes = [node.copy() for node in self.nodes]
            DEPS = [[] for _ in self.nodes]
            
            for arc in sorted(arcs):
                for rel in arc.REL.split('|'): # collapsed repeated arcs
                    DEPS[arc.DEP-1].append(f'{arc.HEAD}:{rel}')
                    
            for node, deps in zip(nodes, DEPS):
                if len(deps) == 0:
                    node.DEPS = '_'
                else:
                    node.DEPS = '|'.join(sorted(deps, key=lambda x: x.split(':')))
            return EnhancedCoNLL.Graph(nodes, arcs, ID=self.ID, annotations=self.annotations)

    @classmethod
    def from_file(cls, path: str) -> CoNLL:
        blocks = re.split('\n{2,}', open(path, 'r').read().strip())
        trees = list(bar(map(EnhancedCoNLL.Graph.from_raw, blocks, range(len(blocks))), total=len(blocks), leave=False, desc=path))
        return cls(trees, path)
    