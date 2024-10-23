from __future__ import annotations
from typing import List, Set, Optional, Union, Tuple, Dict 
import numpy as np 
import torch 

from separ.utils import parallel, flatten
from separ.structs import AbstractNode, Graph, Arc, AbstractDataset, has_cycles, cycles, adjacent_from_arcs


class CoNLL(AbstractDataset):
    """Representation of the CoNLL-U format (https://universaldependencies.org/format.html)"""

    EXTENSION = 'conllu'
    SEP = '\n\n'
    END = '\n\n'
    
    class Node(AbstractNode):
        """Abstract implementation of a node in a Dependency Graph."""
        FIELDS = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        
        def __init__(
            self, 
            ID: Union[int, str], 
            FORM: str, 
            LEMMA: str, 
            UPOS: str, 
            XPOS: str, 
            FEATS: str, 
            HEAD: Optional[Union[str, int]], 
            DEPREL: str, 
            DEPS: str, 
            MISC: str
        ):
            super().__init__(int(ID))
            self.FORM = FORM 
            self.LEMMA = LEMMA 
            self.UPOS = UPOS 
            self.XPOS = XPOS 
            self.FEATS = FEATS
            self.HEAD = int(HEAD) if HEAD is not None else None 
            self.DEPREL = DEPREL
            self.DEPS = DEPS 
            self.MISC = MISC 
            
        @classmethod
        def from_raw(cls, line: str) -> CoNLL.Node:
            """Creates a node from CoNLL raw line.

            Args:
                line (str): Input line.

            Returns:
                CoNLL.Node: Built CoNLL node instance.
                
            Examples:
            >>> node = CoNLL.Node.from_raw('1	Al	Al	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No')
            >>> node.ID, node.FORM
            (1, 'Al')
            """
            return CoNLL.Node(*line.split(cls.SEP))
        
    class Graph(Graph):
        """Abstract implementation of a Dependency Tree."""
        
        def __init__(
            self, 
            nodes: List[CoNLL.Node], 
            arcs: List[Arc], 
            annotations: List[Union[int, str]] = [], 
            ID: Optional[int] = None
        ):
            # one-headed, acyclic and unique root restrictions 
            assert len(nodes) == len(arcs), f'Number of arcs and number of nodes must be the same\n#nodes={len(nodes)}\n#arcs={len(arcs)}'
            assert len(nodes) == 0 or sum(arc.HEAD == 0 for arc in arcs) == 1, f'Graph has more than one root\n{nodes}\n{arcs}'
            assert not has_cycles(adjacent_from_arcs(arcs, len(nodes))), f'Graph has cycles\n' + '\n'.join(node.format() for node in nodes) + f'\n{adjacent_from_arcs(arcs, len(nodes))}'
            self.NODE = CoNLL.Node 
            super().__init__(nodes, arcs, annotations, ID)
                
        def rebuild(self, new_arcs: List[Arc], default: str = 'punct') -> CoNLL.Graph:
            nodes = [node.copy() for node in self.nodes]
            # null all nodes
            for node in nodes:
                node.HEAD = None 
                node.DEPREL = None 
            new_arcs = sorted(new_arcs)
            for arc, node in zip(new_arcs, nodes):
                node.HEAD = arc.HEAD 
                node.DEPREL = arc.REL 
            for node in nodes:
                if node.HEAD is None:
                    node.HEAD = node.ID - 1
                    node.DEPREL = default
            return CoNLL.Graph(nodes, new_arcs, self.annotations, self.ID)
        
        def rebuild_tags(self, tags: List[str]) -> CoNLL.Graph:
            nodes = []
            for node, tag in zip(self.nodes, tags):
                new = node.copy()
                new.UPOS = tag 
                nodes.append(new)
            return CoNLL.Graph(nodes, self.arcs, self.annotations, self.ID)
        
        def greedy_2planar(self) -> Tuple[Set[Arc], Set[Arc]]:
            """2-planar greedy assignment, from https://aclanthology.org/2020.coling-main.223.pdf.

            Returns:
                Tuple[Set[Arc], Set[Arc]]: First and second planes.
            """
            plane1, plane2 = set(), set()
            for r in range(0, len(self) + 1):
                for l in range(r-1, -1, -1):
                    try:
                        next_arc = [arc for arc in self.arcs if arc.left == l and arc.right == r].pop(0)
                        C = set(arc for arc in plane1 | plane2 if arc.cross(next_arc))
                        if len(C & plane1) == 0:
                            plane1.add(next_arc)
                        elif len(C & plane2) == 0:
                            plane2.add(next_arc)
                    except IndexError:
                        continue 
            return plane1, plane2 
        
        def propagate_2planar(self) -> Tuple[Set[Arc], Set[Arc]]:
            """2-planar propagation assignment, from https://aclanthology.org/2020.coling-main.223.pdf.
            
            Returns:
                Tuple[Set[Arc], Set[Arc]]: First and second planes.
            """
            
            plane1, plane2, _plane1, _plane2 = set(), set(), set(), set()
            for r in range(0, len(self) + 1):
                for l in range(r-1, -1, -1):
                    try:
                        next_arc = [arc for arc in self.arcs if arc.left == l and arc.right == r].pop(0)
                        if next_arc not in _plane1:
                            plane1.add(next_arc)
                            propagate(self.arcs, _plane1, _plane2, next_arc, 2)
                        elif next_arc not in _plane2:
                            plane2.add(next_arc)
                            propagate(self.arcs, _plane1, _plane2, next_arc, 1)
                    except IndexError:
                        continue 
            return plane1, plane2 
        
        def planarize(self, k: int) -> CoNLL.Graph:
            """Returns a k-planar version of the current dependency graph.

            Args:
                k (int): Number of planes to retrieve.

            Returns:
                CoNLL.Graph: k-planarized graph.
            """
            if len(self.planes) <= k:
                return self.copy()
            else:
                planes = {0: [self.arcs[self.root.ID-1]]}
                queue = [self.root.ID]
                while len(queue) > 0:
                    head = queue.pop(0)
                    deps = [arc for arc in self.arcs if arc.HEAD == head]
                    for arc in deps:
                        assigned = False 
                        for p in range(k):
                            if p not in planes.keys():
                                planes[p] = [arc]
                                assigned = True 
                                break
                            elif not any(arc.cross(other) for other in planes[p]):
                                planes[p].append(arc)
                                assigned = True 
                                break 
                        h = arc.HEAD
                        while not assigned:
                            opt = Arc(self.arcs[h-1].HEAD, arc.DEP, arc.REL)
                            for p in range(k):
                                if p not in planes.keys():
                                    planes[p] = [opt]
                                    assigned = True 
                                    break 
                                elif not any(opt.cross(other) for other in planes[p]):
                                    planes[p].append(opt)
                                    assigned = True 
                                    break
                            h = self.arcs[h-1].HEAD
                        queue.append(arc.DEP)
                return self.rebuild(sorted(flatten(*planes.values())))

        @property
        def root(self) -> CoNLL.Node:
            return [node for node in self.nodes if node.HEAD == 0].pop()
        
        @property 
        def heads(self) -> np.ndarray:
            return np.array(self.HEAD)
        
        @property
        def rels(self) -> np.ndarray:
            return np.array(self.DEPREL)
        
        @property
        def tags(self) -> List[str]:
            return [node.UPOS for node in self.nodes]
        
        @classmethod 
        def from_raw(cls, lines: str) -> CoNLL.Graph:
            annotations, nodes, arcs = [], [], []
            for line in lines.strip().split(cls.SEP):
                if not line.split()[0].isdigit():
                    annotations.append(line)
                else:
                    annotations.append(len(nodes))
                    node = CoNLL.Node.from_raw(line)
                    nodes.append(node)
                    arcs.append(Arc(node.HEAD, node.ID, node.DEPREL))
            return cls(nodes, arcs, annotations)
        
        @property
        def planes(self) -> Dict[int, List[Arc]]:
            """Greedy plane assignment (from left to right sorting by the dependant node). The planes 
            are sorted by the number of arcs or the creation order and the first plane always contains 
            the root node.
            
            Returns:
                Dict[int, List[Arc]]: Plane assignment.
            """
            if self._planes is None:
                # store the planes 
                self._planes = {0: []}
                for arc in sorted(self.arcs, key=lambda arc: arc.left):
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
                Dict[int, List[Arc]]: Relaxed plane assignment.
            """
            if self._relaxed_planes is None:
                self._relaxed_planes = {0: []}
                for arc in sorted(self.arcs, key=lambda arc: arc.left):
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
        
        def rebuild_field(self, field: str, values: List[str]) -> CoNLL.Graph:
            assert field not in ['HEAD', 'DEPREL'], 'Arcs cannot depend on the fields to rebuild'
            nodes = self.nodes.copy()
            for value, node in zip(values, nodes):
                node.__setattr__(field, value)
            return CoNLL.Graph(nodes, self.arcs, self.annotations, self.ID)
            
    @classmethod 
    def from_file(cls, path: str, num_workers: int = 1) -> CoNLL:
        blocks = list(filter(lambda x: len(x) > 0, open(path, 'r').read().strip().split('\n\n')))
        graphs = parallel(cls.Graph.from_raw, blocks, num_workers=num_workers, name=path.split('/')[-1])
        return cls(graphs, path)
    
    
def propagate(arcs: List[Arc], _plane1: Set[Arc], _plane2: Set[Arc], arc: Arc, i: int) -> Tuple[Set[Arc], Set[Arc]]:
    selected = _plane1 if i == 1 else _plane2
    no_selected = _plane1 if i == 2 else _plane2
    selected.add(arc)
    for other in arcs:
        if arc.cross(other) and other not in no_selected:
            _plane1, _plane2 = propagate(arcs, _plane1, _plane2, other, 3-i)
    return _plane1, _plane2

        
class EnhancedCoNLL(CoNLL):
    """Representation of eh Enhanced CoNLL-U format for Semantic Dependency Parsing 
    (https://universaldependencies.org/v2/conll-u.html)"""
            
    class Graph(Graph):
        def __init__(self, nodes: List[CoNLL.Node], arcs: List[Arc], annotations: List[Union[str, int]] = [], ID: Optional[int] = None):
            self.NODE = CoNLL.Node
            super().__init__(nodes, arcs, annotations, ID)

        @classmethod
        def from_raw(cls, lines: str) -> EnhancedCoNLL.Graph:
            lines = list(filter(lambda x: len(x) > 0, lines.strip().split('\n')))
            nodes, annotations = [], []
            for line in lines:
                if line.split()[0].isdigit():
                    annotations.append(len(nodes))
                    nodes.append(CoNLL.Node.from_raw(line))
                elif line.startswith('#'):
                    annotations.append(line)
            arcs = []
            for i, node in enumerate(nodes):
                if node.DEPS == '_':
                    continue 
                values = node.DEPS.split('|')
                _arcs = []
                for value in values:
                    head, *rel = value.split(':')
                    rel = ':'.join(rel)
                    if head.isdigit() and int(head) != node.ID and rel != '_': # avoid cycles
                        _arcs.append(Arc(int(head), node.ID, rel))
                arcs += _arcs
                node.HEAD = i # all nodes with HEAD to ID - 1
                if len(_arcs) == 0:
                    node.DEPS = '_'
                else:
                    node.DEPS = '|'.join(f'{arc.HEAD}:{arc.REL}' for arc in _arcs)
            return cls(nodes, arcs, annotations)
        
        def rebuild(self, arcs: List[Arc]) -> EnhancedCoNLL.Graph:
            # assert len(set(node.ID for node in self.nodes)) == len(set(arc.DEP for arc in arcs)), f'There is some node without heads\n{arcs}'
            nodes = [node.copy() for node in self.nodes]
            DEPS = [[] for _ in self.nodes]
            
            for arc in sorted(arcs):
                DEPS[arc.DEP-1].append(f'{arc.HEAD}:{arc.REL}')
            
            for node, deps in zip(nodes, DEPS):
                if len(deps) == 0:
                    node.DEPS = '_'
                else:
                    node.DEPS = '|'.join(deps)
            return EnhancedCoNLL.Graph(nodes, arcs, self.annotations, self.ID)
    
    
def MST(scores: torch.Tensor) -> torch.Tensor:
    """Maximum-Spanning Tree algorithm from https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf.

    Args:
        scores (torch.Tensor): Input score matrix.

    Returns:
        torch.Tensor: Adjacent matrix from a dependency tree.
    """
    scores = preprocess(scores)
    n = scores.shape[0]
    heads = scores.argmax(-1)
    adjacent = torch.zeros(n, n, dtype=torch.bool)
    adjacent[list(range(n)), heads] = True 
    
    for cycle in cycles(adjacent):
        if cycle != {0}:
            scores = contract(scores - scores.max(-1).values.unsqueeze(-1), sorted(cycle))
            new = MST(scores[:, :, 0])
            adjacent = expand(scores, new, adjacent)
            break 
    adjacent[0, :] = False
    return adjacent

def contract(scores: torch.Tensor, cycle: List[int]) -> torch.Tensor:
    """Contract operation as described in https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf.

    Args:
        scores (torch.Tensor): Matrix of scores.
        cycle (List[int]): Nodes of the detected cycle.

    Returns:
        torch.Tensor: Contract indexed score matrix of the reduced graph.
    """
    n = scores.shape[0]
    indices = torch.arange(n).unsqueeze(-1).repeat(1,n)
    scores = torch.stack([scores, indices, indices.T], -1)
    pivot = min(cycle)
    
    # update rows 
    for head in range(n):
        view = scores[cycle, head, 0]
        view = torch.where(view == 0, view.min(), view)
        dep = cycle[view.argmax()]
        scores[pivot, head] = scores[dep, head]
        
    # update cols
    for dep in range(n):
        view = scores[dep, cycle, 0]
        view = torch.where(view == 0, view.min(), view)
        head = cycle[view.argmax()]
        scores[dep, pivot] = scores[dep, head]
        
    cycle.remove(pivot)
    maintain = [i for i in range(n) if i not in cycle]
    return scores[maintain][:, maintain]

def expand(scores: torch.Tensor, adjacent: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Expand operation as described in https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf.

    Args:
        scores (torch.Tensor ~ [seq_len, seq_len, 3]): Indexed matrix of scores (score, dep, head).
        adjacent (torch.Tensor): Estimated adjacent matrix.
        prev (torch.Tensor): Previous expanded adjacent matrix.

    Returns:
        torch.Tensor: New expanded matrix.
    """
    expanded = torch.zeros_like(prev, dtype=torch.bool)
    for idep, ihead in adjacent.nonzero():
        _, dep, head = scores[idep, ihead]
        expanded[dep, head] = True 
    for dep in range(prev.shape[0]):
        if expanded[dep].sum() == 0:
            expanded[dep] = prev[dep]
    return expanded

def preprocess(scores: torch.Tensor) -> torch.Tensor:
    """Preprocess a scored adjacent matrix to fulfill the dependency-tree constraints.

    Args:
        scores (torch.Tensor): Scores of the adjacent matrix.

    Returns:
        torch.Tensor: Preprocessed matrix.
    """
    n = scores.shape[0]
    # suppress diagonal 
    scores[list(range(n)), list(range(n))] = scores.min()-1
    
    # only one root
    root = scores[:, 0].argmax(-1)
    scores[:, 0] = scores.min()-1
    scores[root, 0] = scores.max()+1
    
    # cycle in w0 -> w0
    scores[0, 0] = scores.max()
    return scores 


