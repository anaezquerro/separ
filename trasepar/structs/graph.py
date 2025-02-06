from __future__ import annotations
from typing import List, Optional, Dict, Union, Set, Iterator
import numpy as np 
import torch, random 

from trasepar.structs.node import AbstractNode
from trasepar.structs.arc import Arc 

class Graph:
    SEP: str = '\n' # separation 
    NODE = None # type of node
   
    def __init__(
        self, 
        nodes: List[AbstractNode], 
        arcs: List[Arc], 
        annotations: List[Union[int, str]] = [], 
        ID: Optional[int] = None
    ):
        """Abstract representation of a graph.

        Args:
            nodes (List[AbstractNode]): List of nodes.
            arcs (List[Arc]): List of arcs.
            annotations (List[str], optional): Previous list of annotations.
            ID (Optional[int], optional): Integer identifier for sorting. Defaults to None (no sorting).
        """
        # assert not any(node.is_bos() for node in nodes), 'Do not introduce artificial nodes'
        self.nodes = nodes 
        self.arcs = arcs

        self.annotations = annotations
        self.ID = ID
        
        # initialize inner attributes 
        self._ADJACENT = None
        self._planes = None
        self._relaxed_planes = None 
        self._bit6k_planes: List[List[Arc]] = None # store planes from 4k-bit encoding
        self._bit4k_planes: List[List[Arc]] = None # store planes from 6k-bit encoding
        self._cycles = None 
        self._transformed = False 
        
    @property
    def ADJACENT(self) -> torch.Tensor:
        if self._ADJACENT is None:
            self._ADJACENT = torch.zeros(len(self)+1, len(self)+1, dtype=torch.bool)
            for arc in self.arcs:
                self._ADJACENT[arc.DEP, arc.HEAD] = True
        return self._ADJACENT
    
    @property 
    def rels(self) -> Set[str]:
        return set(arc.REL for arc in self.arcs)
    
    def __getattr__(self, field: str):
        if field in self.NODE.FIELDS and field != 'ID':
            return [getattr(node, field) for node in self.nodes]
        else:
            raise AttributeError(field)
        
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __lt__(self, other: Graph) -> bool:
        if isinstance(other, Graph):
            return self.ID < other.ID 
        else:
            raise NotImplementedError
        
    def __getitem__(self, index: int) -> AbstractNode:
        """Returns the node at a given position.
        - Position 0 is reserved for the artificial BOS node.
        - Position i=1..n are reserved for the node i.
        - Position n+1 is reserved for the artificial EOS node.

        Args:
            index (int): Position of the node.

        Returns:
            AbstractNode: _description_
        """
        if index == 0:
            return self.NODE.bos()
        elif index == len(self) + 1:
            return self.NODE.eos(len(self)+1)
        else:
            return self.nodes[index-1]
        
    def __eq__(self, other: Graph):
        return (len(other.nodes) == len(self.nodes)) and (len(other.arcs) == len(self.arcs)) and \
            all(node1 == node2 for node1, node2 in zip(sorted(self.nodes), sorted(other.nodes))) and \
            all(arc1 == arc2 for arc1, arc2 in zip(sorted(self.arcs), sorted(other.arcs)))
        
    def format(self) -> str:
        """Formatted representation of the graph."""
        return self.SEP.join(self.nodes[item].format() if isinstance(item, int) else item for item in self.annotations)
    
    def copy(self) -> Graph:
        return self.__class__([node.copy() for node in self.nodes], [arc.copy() for arc in self.arcs], self.annotations, self.ID)
    
    @property
    def planes(self) -> Dict[int, List[Arc]]:
        """Greedy plane assignment (from left to right sorting by the dependant node). The planes 
        are sorted by the number of arcs or the creation order.

        Returns:
            Dict[int, List[Arc]]: Plane assignment.
        """
        if self._planes is None:
            # store the planes 
            self._planes = {0: []}
            for arc in self.arcs:
                added = False
                for plane in range(len(self._planes)):
                    if not any(arc.cross(other) for other in self._planes[plane]):
                        self._planes[plane].append(arc)
                        added = True 
                        break 
                if not added:
                    self._planes[len(self._planes)] = [arc]
            # sort and locate in the planes with the most number of arcs 
            order = sorted(self._planes.keys(), key=lambda i: (len(self._planes[i]), -i), reverse=True)
            self._planes = {i: self._planes[plane] for i, plane in enumerate(order)}
        return self._planes
    
    @property
    def relaxed_planes(self) -> Dict[int, List[Arc]]:
        """Greedy relaxed plane assignment (from left to right sorting by the dependant node).

        Returns:
            Dict[int, List[Arc]]: Relaxed plane assigment.
        """
        if self._relaxed_planes is None:
            self._relaxed_planes = {0: []}
            for arc in self.arcs:
                added = False
                for plane in range(len(self._relaxed_planes)):
                    if not any(arc.cross(other) and arc.side == other.side for other in self._relaxed_planes[plane]):
                        self._relaxed_planes[plane].append(arc)
                        added = True 
                        break 
                if not added:
                    self._relaxed_planes[len(self._relaxed_planes)] = [arc]
            # sort and locate in the planes with the most number of arcs 
            order = sorted(self._relaxed_planes.keys(), key=lambda i: (len(self._relaxed_planes[i]), -i), reverse=True)
            self._relaxed_planes = {i: self._relaxed_planes[plane] for i, plane in enumerate(order)}
        return self._relaxed_planes
    
    @property 
    def LABELED_ADJACENT(self) -> np.ndarray:
        """Labeled adjacent matrix representation of the graph.

        Returns:
            np.ndarray: Labeled matrix where each cell [i,j] contains the label of the arc from  
                node j to node i, or the symbol _ if there is no arc between nodes i and j.
        """
        graph = np.empty((len(self)+1, len(self)+1), np.object_)
        graph[:] = '_'
        for arc in self.arcs:
            graph[arc.DEP, arc.HEAD] = arc.REL 
        return graph 
    
    @property 
    def n_in(self) -> List[int]:
        """Number of incoming arcs (number of heads per node).

        Returns:
            List[int]: Number of incoming arcs for each node.
        """
        n_in = [0 for _ in range(len(self))]
        for arc in self.arcs:
            n_in[arc.DEP-1] += 1
        return n_in 
    
    @property
    def n_out(self) -> List[int]:
        """Number of outcoming arcs (number of dependants per node).

        Returns:
            List[int]: Number of outcoming arcs for each node.
        """
        n_out = [0 for _ in range(len(self)+1)]
        for arc in self.arcs:
            n_out[arc.HEAD] += 1
        return n_out

    @property 
    def n_left_in(self) -> List[int]:
        """Number of left incoming arcs (number of left heads per node).

        Returns:
            List[int]: Number of left heads per node.
        """
        n_left_in = [0 for _ in range(len(self))]
        for arc in self.arcs:
            if arc.HEAD < arc.DEP: 
                n_left_in[arc.DEP-1] += 1
        return n_left_in 
    
    @property
    def n_right_in(self) -> List[int]:
        """Number of right incoming arcs (number of right heads per node).

        Returns:
            List[int]: Number of right hedas per node.
        """
        n_right_in = [0 for _ in range(len(self))]
        for arc in self.arcs:
            if arc.DEP < arc.HEAD:
                n_right_in[arc.DEP-1] += 1
        return n_right_in
                
    @property 
    def n_cycles(self) -> int:
        return len(self.cycles)
    
    @property 
    def cycles(self) -> List[Set[int]]:
        if self._cycles is not None:
            return self._cycles
        self._cycles = list(cycles(self.ADJACENT))
        return self._cycles
    
    def has_cycles(self) -> bool:
        return has_cycles(self.ADJACENT)
    
    def to_tikz(self) -> str:
        latex = r'\begin{dependency}' + '\n\t' + r'\begin{deptext}' + '\n\t\t'
        # add words
        latex += '\& '.join(node.FORM for node in self.nodes)
        latex += r'\\'  + '\n\t' + r'\end{deptext}' + '\n\t'
        # add edges 
        latex +=  '\n\t'.join(r'\depedge{' + str(arc.HEAD) + '}{' + str(arc.DEP) + '}{' + arc.REL + '}' for arc in self.arcs if arc.HEAD != 0) + '\n\t'
        latex += '\n\t'.join(r'\deproot{' + str(arc.DEP) + '}{' + str(arc.REL) + '}' for arc in self.arcs if arc.HEAD == 0) + '\n'
        latex += r'\end{dependency}'
        return latex
        
    
        
        
def has_cycles(adjacent: torch.Tensor) -> bool:
    """Checks whether the adjacent matrix contains cycles.

    Args:
        adjacent (torch.Tensor ~ [seq_len, seq_len]): Adjacent matrix of the graph.

    Returns:
        bool: Whether the graph has cycles.
    """
    n_in = adjacent.sum(-1)    
    # enqueue vertices with 0 in-degree
    queue = (n_in == 0).nonzero().flatten().tolist()
    
    visited = 0
    while len(queue) > 0:
        node = queue.pop(0)
        visited += 1
        for neighbor in adjacent[:, node].nonzero().flatten().tolist():
            n_in[neighbor] -= 1
            if n_in[neighbor] == 0:
                queue.append(neighbor)
    return visited != adjacent.shape[0]

def cycles(
    adjacent: torch.Tensor, 
    _visited: Optional[List[int]] = None,
    _non_visited: Optional[Set[int]] = None,
    _recovered: Optional[List[Set[int]]] = None
) -> Iterator[Set[int]]:
    """Obtain cycles from an adjacent matrix.

    Args:
        adjacent (torch.Tensor ~ [seq_len, seq_len]): Adjacent matrix of the graph.
        _visited (Optional[List[int]], optional): Visited nodes. Defaults to None.
        _non_visited (Optional[Set[int]], optional): Non visited nodes. Defaults to None.
        _recovered (Optional[List[Set[int]]], optional): Recovered cycles. Defaults to None.

    Returns:
        Set[int]: Detected cycle.

    Yields:
        Iterator[Set[int]]: Iterable of cycles.
    """
    if _visited is None:
        _visited = [0]
    if _non_visited is None:
        _non_visited = set(i for i in range(adjacent.shape[0]))
    if _recovered is None:
        _recovered = []
    head = _visited[-1]
    _non_visited -= {head}
    deps = adjacent[:, head].nonzero().flatten().tolist()
    for dep in deps:
        _non_visited -= {dep}
        if dep in _visited:
            cycle = sorted(set(_visited[_visited.index(dep):]))
            if cycle not in _recovered:
                yield cycle 
                _recovered.append(cycle)
        else:
            for cycle in cycles(adjacent, _visited + [dep], _non_visited, _recovered):
                if cycle is not None:
                    yield cycle 
    if len(_non_visited) > 0:
        head = sorted(_non_visited)[0]
        _non_visited.remove(head)
        for cycle in cycles(adjacent, [head], _non_visited, _recovered):
            yield cycle 
    pass 

def forms_cycles(adjacent: torch.Tensor, dep: int, head: int) -> bool:
    new = adjacent.clone()
    new[dep, head] = True 
    return has_cycles(new)

def adjacent_from_arcs(arcs: List[Arc], n: int) -> torch.Tensor:
    adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
    for arc in arcs:
        adjacent[arc.DEP, arc.HEAD] = True 
    return adjacent 

def candidates_no_cycles(adjacent: torch.Tensor, dep: int) -> List[int]: 
    n = adjacent.shape[0]
    candidates = []
    for o in range(1, max(n-dep, dep)+1):
        for side in (-1, 1):
            head = dep+side*o
            if head in range(n) and (head != 0 or not adjacent[:, 0].any()) and not forms_cycles(adjacent, dep, head):
                candidates.append(head)
    return candidates
                