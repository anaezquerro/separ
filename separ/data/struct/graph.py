from __future__ import annotations
from typing import Iterator
import numpy as np 
import torch, re

from separ.data.struct.sentence import Sentence  
from separ.data.struct.arc import Arc 

class Graph(Sentence):
    def __init__(
        self, 
        nodes: list[Sentence.Node], 
        arcs: list[Arc], 
        ID: Optional[int] = None,
        annotations: Optional[list[Union[int, str]]] = None
    ):
        """Abstract representation of a graph, which adds a list of arcs to a sentence..

        Args:
            nodes (list[Node]): List f nodes..
            arcs (list[Arc]): List of arcs.
            ID (Optional[int], optional): Integer identifier for sorting. Defaults to None (no sorting).
            annotations (Optional[list[Union[str, int]]]): List of annotations or node position.
        """
        super().__init__(nodes, ID, annotations)
        self.arcs = arcs
        
        # initialize inner attributes 
        self._ADJACENT = None
        self._planes = None
        self._relaxed_planes = None 
        self._bit6k_planes: list[list[Arc]] = None # store planes from 4k-bit encoding
        self._bit4k_planes: list[list[Arc]] = None # store planes from 6k-bit encoding
        self._cycles = None 
        self.transformed = False 
        

        
    def rebuild_from_arcs(self, new_arcs: list[Arc]) -> Graph:
        raise NotImplementedError
        
    @property
    def ADJACENT(self) -> torch.Tensor:
        if self._ADJACENT is None:
            self._ADJACENT = torch.zeros(len(self)+1, len(self)+1, dtype=torch.bool)
            for arc in self.arcs:
                self._ADJACENT[arc.DEP, arc.HEAD] = True
        return self._ADJACENT
    
    @property 
    def rels(self) -> set[str]:
        return set(arc.REL for arc in self.arcs)
    
    @property 
    def n_roots(self) -> int:
        return sum(arc.HEAD == 0 for arc in self.arcs)
    
    def __eq__(self, other: Graph):
        return len(self) == len(other) and \
            all(node1 == node2 for node1, node2 in zip(self.nodes, other.nodes)) and \
            all(arc1 == arc2 for arc1, arc2 in zip(sorted(self.arcs), sorted(other.arcs)))
        
    def copy(self) -> Graph:
        return self.__class__([node.copy() for node in self.nodes], [arc.copy() for arc in self.arcs], self.ID)
    
    @property
    def planes(self) -> dict[int, list[Arc]]:
        """Greedy plane assignment (from left to right sorting by the dependant node). The planes 
        are sorted by the number of arcs or the creation order.

        Returns:
            dict[int, list[Arc]]: Plane assignment.
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
    def relaxed_planes(self) -> dict[int, list[Arc]]:
        """Greedy relaxed plane assignment (from left to right sorting by the dependant node).

        Returns:
            dict[int, list[Arc]]: Relaxed plane assigment.
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
    def bit4k_planes(self) -> list[list[Arc]]:
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
    def bit6k_planes(self) -> list[list[Arc]]:
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
    def HEAD(self) -> list[list[int]]:
        """Returns the list of parents positions for each word.

        Returns:
            list[list[int]]: List of parents positions for each word in the graph.
        """
        heads = [[] for _ in range(len(self))]
        for arc in self.arcs:
            heads[arc.DEP-1].append(arc.HEAD)
        return heads
    
    @property 
    def DEPS(self) -> list[list[int]]:
        """Returns the list of dependents of each node [0,n].

        Returns:
            list[list[int]]: List of dependents per node.
        """
        deps = [[] for _ in range(len(self) + 1)]
        for arc in self.arcs:
            deps[arc.HEAD].append(arc.DEP)
        return deps
    
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
    def n_in(self) -> list[int]:
        """Number of incoming arcs (number of heads per node).

        Returns:
            list[int]: Number of incoming arcs for each node.
        """
        n_in = [0 for _ in range(len(self))]
        for arc in self.arcs:
            n_in[arc.DEP-1] += 1
        return n_in 
    
    @property
    def n_out(self) -> list[int]:
        """Number of outcoming arcs (number of dependants per node).

        Returns:
            list[int]: Number of outcoming arcs for each node.
        """
        n_out = [0 for _ in range(len(self)+1)]
        for arc in self.arcs:
            n_out[arc.HEAD] += 1
        return n_out

    @property 
    def n_left_in(self) -> list[int]:
        """Number of left incoming arcs (number of left heads per node).

        Returns:
            list[int]: Number of left heads per node.
        """
        n_left_in = [0 for _ in range(len(self))]
        for arc in self.arcs:
            if arc.HEAD < arc.DEP: 
                n_left_in[arc.DEP-1] += 1
        return n_left_in 
    
    @property
    def n_right_in(self) -> list[int]:
        """Number of right incoming arcs (number of right heads per node).

        Returns:
            list[int]: Number of right hedas per node.
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
    def cycles(self) -> list[set[int]]:
        if self._cycles is not None:
            return self._cycles
        self._cycles = list(cycles(self.ADJACENT))
        return self._cycles
    
    def to_tikz(self) -> str:
        latex = r'\begin{dependency}' + '\n\t' + r'\begin{deptext}' + '\n\t\t'
        # add words
        latex += r'\& '.join(node.FORM for node in self.nodes)
        latex += r'\\'  + '\n\t' + r'\end{deptext}' + '\n\t'
        # add edges 
        latex +=  '\n\t'.join(r'\depedge{' + str(arc.HEAD) + '}{' + str(arc.DEP) + '}{' + arc.REL + '}' for arc in self.arcs if arc.HEAD != 0) + '\n\t'
        latex += '\n\t'.join(r'\deproot{' + str(arc.DEP) + '}{' + str(arc.REL) + '}' for arc in self.arcs if arc.HEAD == 0) + '\n'
        latex += r'\end{dependency}'
        return latex
        
    def collapse_one_cycles(self) -> Graph:
        """Collapses cycles of length one to create a dependency that goes from the previous node to the actual node.
        To identify the 1-length cycle, the special <c></c> is used for the relation. """
        graph = self.copy()
        for arc in graph.arcs:
            if arc.is_cycle():
                arc.HEAD = arc.DEP-1 
                arc.REL = f'<c>{arc.REL}</c>'
        return graph.rebuild_from_arcs(graph.arcs)
    
    @classmethod
    def decollapse_one_cycles(cls, arcs: list[Arc]) -> Graph:
        """Decollapses cycles of length one by creating a cycle for those arcs whose REL has the expression <c></c>."""
        for arc in arcs:
            matches = re.findall(r'<c>(.*?)</c>', arc.REL)
            if len(matches) > 0:
                arc.HEAD = arc.DEP 
                arc.REL = matches[0]
        return arcs
    
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
    _visited: list[int] | None = None,
    _non_visited: set[int] | None = None,
    _recovered: list[set[int]] | None = None
) -> Iterator[set[int]]:
    """Obtain cycles from an adjacent matrix.

    Args:
        adjacent (torch.Tensor ~ [seq_len, seq_len]): Adjacent matrix of the graph.
        _visited (Optional[list[int]], optional): Visited nodes. Defaults to None.
        _non_visited (Optional[set[int]], optional): Non visited nodes. Defaults to None.
        _recovered (Optional[list[set[int]]], optional): Recovered cycles. Defaults to None.

    Returns:
        set[int]: Detected cycle.

    Yields:
        Iterator[set[int]]: Iterable of cycles.
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
    if head == dep:
        return True 
    elif not adjacent[head].any():
        return False 
    else:
        head_heads = adjacent[head].nonzero().flatten().tolist() 
        if len(head_heads) > 1:
            return any(forms_cycles(adjacent, dep, h) for h in head_heads)
        else:
            return forms_cycles(adjacent, dep, head_heads[0])

def adjacent_from_arcs(arcs: list[Arc], n: int) -> torch.Tensor:
    adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
    for arc in arcs:
        adjacent[arc.DEP, arc.HEAD] = True 
    return adjacent 

def candidates_no_cycles(adjacent: torch.Tensor, dep: int) -> list[int]: 
    n = adjacent.shape[0]
    candidates = []
    for o in range(1, max(n-dep, dep)+1):
        for side in (-1, 1):
            head = dep+side*o
            if head in range(n) and (head != 0 or not adjacent[:, 0].any()) and not forms_cycles(adjacent, dep, head):
                candidates.append(head)
    return candidates
                