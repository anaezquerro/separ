from __future__ import annotations
from typing import List, Set, Optional, Union, Tuple, Dict 
import numpy as np 
import os, shutil

from trasepar.utils import parallel, flatten, shell, folderpath, filename, mkdtemp
from trasepar.structs import AbstractNode, Graph, Arc, AbstractDataset, has_cycles, adjacent_from_arcs, candidates_no_cycles
from trasepar.data.sdp import SDP 
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
        
        def rebuild_field(self, field: str, values: List[str]) -> CoNLL.Graph:
            """Updates an entire field of the graph.

            Args:
                field (str): Field of the graph.
                values (List[str]): Values to update.

            Returns:
                CoNLL.Graph: Updated graph.
            """
            assert field not in ['HEAD', 'DEPREL'], 'Arcs cannot depend on the fields to rebuild'
            assert len(values) == len(self.nodes), f'Length mismatch between nodes and values: {len(values)} != {len(self.nodes)}'
            nodes = self.nodes.copy()
            for value, node in zip(values, nodes):
                node.__setattr__(field, value)
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
            """Returns a k-planar version of the current dependency graph. This algorithm starts in the root 
            of the dependency tree and makes a BFS adding arcs to the k planes. When it is not possible to 
            assign a new arc to any plane, it changes its head by the head of its head itertively until it 
            is assigned to one plane.
            
            Args:
                k (int): Number of planes to retrieve.

            Returns:
                CoNLL.Graph: k-planar graph.
            """
            if len(self.planes) <= k:
                return self.copy()
            else:
                queue = [0]
                arcs = [arc.copy() for arc in self.arcs]
                rec = {p: [] for p in range(k)}
                visited = set()
                while len(queue) > 0:
                    head = queue.pop(0)
                    deps = [arc for arc in arcs if arc.HEAD == head]
                    for dep in deps:
                        assigned = False 
                        for plane in rec.values():
                            if not any(dep.cross(arc) for arc in plane):
                                plane.append(dep)
                                assigned = True 
                                break 
                        # force assignment 
                        while not assigned and arcs[dep.HEAD-1].HEAD != 0:
                            dep.HEAD = arcs[dep.HEAD-1].HEAD
                            planes = [plane for plane in rec.values() if not any(dep.cross(arc) for arc in plane)]
                            if len(planes) > 0:
                                planes.pop(0).append(dep)
                                assigned = True 
                                break 
                        if not assigned:
                            candidates = sorted(visited, key=lambda x: abs(x-dep.DEP))
                            while not assigned:
                                dep.HEAD = candidates.pop(0)
                                planes = [plane for plane in rec.values() if not any(dep.cross(arc) for arc in plane)]
                                if len(planes) > 0:
                                    planes.pop(0).append(dep)
                                    assigned = True 
                                    break 
                        queue.append(dep.DEP)
                        visited.add(dep.DEP)
                return self.rebuild(arcs)
            
        def distribute_projective(self) -> List[CoNLL.Graph]:
            """Distributes the arcs of the non-projective dependency tree in projective trees.

            Returns:
                List[CoNLL.Graph]: List of projective trees.
            """
            
            def covers_head(arc: Arc, plane: List[Arc]):
                """Checks whether a dependant arc covers a parent in its branch.

                Args:
                    arc (Arc): Evaluated arc.
                    plane (List[Arc]): List of arcs of a plane. 
                """
                heads = set(other.HEAD for other in plane) - set(other.DEP for other in plane)
                return any(arc.left < head < arc.right for head in heads)
            
            queue = [0]
            trees = [[]]
            active = 0
            while len(queue) > 0:
                head = queue.pop(0)
                deps = [arc for arc in self.arcs if arc.HEAD == head]
                for dep in deps:
                    added = False 
                    while not added:
                        if active >= len(trees):
                            trees.append([dep])
                            added = True
                        elif not any(map(dep.cross, trees[active])) and not covers_head(dep, trees[active]):
                            trees[active].append(dep)
                            active = 0
                            added = True 
                        else:
                            active += 1
                    queue.append(dep.DEP)
            return [self.fill_projective(tree) for tree in trees]
        
        def fill_projective(self, arcs: List[Arc], NULL: str = '<none>') -> List[CoNLL.Graph]:
            # create arcs that are inside ropes
            arcs = sorted(arcs, key=len) 
            for arc in arcs:
                for dep in range(arc.left+1, arc.right):
                    if dep not in set(arc.DEP for arc in arcs):
                        arcs.append(Arc(arc.left if arc.left != 0 else arc.right, dep, NULL))
            deps = set(arc.DEP for arc in arcs)
            heads = [i for i in range(1, len(self)+1) if i not in deps]
            root = [arc.DEP for arc in arcs if arc.HEAD == 0]
            if len(root) > 0:
                arcs += [Arc(root[0], head, NULL) for head in heads]
            else:
                for i, head in enumerate(heads):
                    arcs.append(Arc(0 if i == 0 else heads[i-1], head, NULL))
            return self.rebuild(sorted(arcs))
                
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
                queue = [0]
                while len(queue) > 0:
                    head = queue.pop(0)
                    deps = [arc for arc in self.arcs if arc.HEAD == head]
                    for dep in deps:
                        added = False 
                        for plane in range(len(self._planes)):
                            if not any(dep.cross(other) for other in self._planes[plane]):
                                self._planes[plane].append(dep)
                                added = True 
                                break 
                        queue.append(dep.DEP)
                        if not added:
                            self._planes[len(self._planes)] = [dep]
                # sort and locate in the planes with the most number of arcs 
                order = [0] + sorted(self._planes.keys() - {0}, key=lambda i: (len(self._planes[i]), -i), reverse=True)
                self._planes = {i: self._planes[plane] for i, plane in enumerate(order)}
            return self._planes
        
        @property
        def relaxed_planes(self) -> Dict[int, List[Arc]]:
            """Greedy relaxed plane assignment (from left to right sorting by the dependant node).
            Relaxed planes allow crossing arcs in different directions.

            Returns:
                Dict[int, List[Arc]]: Relaxed plane assignment.
            """
            if self._relaxed_planes is None:
                self._relaxed_planes = {0: []}
                for arc in sorted(self.arcs, key=lambda arc: arc.left):
                    added = False
                    for plane in range(len(self._relaxed_planes)):
                        # only add the arc in this plane if it does not cross any other in the same direction
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
        
        def is_projective(self) -> bool:
            return len(self.planes) == 1
       
        def projectivize(self, mode: str = 'head') -> Graph.CoNLL:
            """Pseudo-projective algorithm from [Nivre and Nilsson, (2005)](https://aclanthology.org/P05-1013/).

            Args:
                mode (str): Mode to encode the lifting operations.

            Returns:
                Graph.CoNLL: Projective dependency tree.
            """
            assert mode in ['head', 'head+path', 'path'], NotImplementedError('The projective mode is not available')
            proj = self.copy() # create a copy that will be projective
            while not proj.is_projective():
                nonp_arc = sorted(flatten(plane for p, plane in proj.planes.items() if p > 0), key=len).pop(0)
                parent = proj.arcs[nonp_arc.HEAD-1] # parent arc
                nonp_arc.HEAD = parent.HEAD 
                # we use ¡ as the up arrow and ! as the down arrow
                if mode == 'head' or mode == 'head+path':
                    nonp_arc.REL = f'{nonp_arc.REL}¡{parent.REL}'
                else:
                    nonp_arc.REL += '¡'
                if mode == 'head+path' or mode == 'path':
                    parent.REL += '!'
                proj = proj.rebuild(proj.arcs)
            return proj 
        
        def deprojectivize(self, mode: str = 'head') -> Graph.CoNLL:
            """Pseudo-deprojective algorithm from [Nivre and Nilsson, (2005)](https://aclanthology.org/P05-1013/).

            Args:
                mode (str): Mode to decode the lifting operations.

            Returns:
                Graph.CoNLL: Non-projective recovered dependency tree.
            """
            assert mode in ['head', 'head+path', 'path'], NotImplementedError('The projective mode is not available')
            nonp = self.copy() # create non-projective copy
            # get non-projective arcs
            nonp_arcs = [arc for arc in nonp.arcs if '¡' in arc.REL and arc.HEAD != 0]
            while len(nonp_arcs) > 0:
                # search a target arc
                nonp_arc = nonp_arcs[0]
                hrel = nonp_arc.REL.split('¡')[-1]
                found = False
                if mode == 'head+path' or mode == 'path':
                    # get arcs where nonp_arc.HEAD = arc.HEAD
                    search = sorted(arc for arc in nonp.arcs if arc.HEAD == nonp_arc.HEAD and arc.DEP != nonp_arc.DEP) 
                    while len(search) > 0:
                        target = search.pop(0)
                        deps = sorted(arc for arc in nonp.arcs if arc.HEAD == target.DEP) 
                        if f'{target.REL}!' == f'{hrel}!' and (mode != 'path' or not any('!' in dep.REL for dep in deps)):
                            nonp_arc.HEAD = target.DEP
                            found = True
                            break 
                        search += deps
                if not found and mode != 'path':
                    search = sorted(arc for arc in nonp.arcs if arc.HEAD == nonp_arc.HEAD and arc.DEP != nonp_arc.DEP) 
                    while len(search) > 0:
                        target = search.pop(0)
                        if target.REL == hrel:
                            nonp_arc.HEAD = target.DEP 
                            break 
                        search += sorted(arc for arc in nonp.arcs if arc.HEAD == target.DEP) 
                nonp_arc.REL = '¡'.join(nonp_arc.REL.split('¡')[:-1])
                if '¡' not in nonp_arc.REL:
                    nonp_arcs.pop(0)
            return nonp.rebuild(nonp.arcs)

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
        
    @classmethod 
    def from_file(cls, path: str, num_workers: int = 1) -> CoNLL:
        blocks = list(filter(lambda x: len(x) > 0, open(path, 'r').read().strip().split('\n\n')))
        graphs = parallel(cls.Graph.from_raw, blocks, num_workers=num_workers, name=path.split('/')[-1])
        return cls(graphs, path)
    
    def projectivize(self, mode: str = 'head', num_workers: int = 1) -> CoNLL:
        proj = parallel(self._proj, self, [mode for _ in self], num_workers=num_workers, name=f'proj-{self.name}')
        path = self.path + f'-proj-{mode}'
        return CoNLL(proj, path=path)
    
    def deprojectivize(self, mode: str = 'head', num_workers: int = 1) -> CoNLL:
        nonp = parallel(self._deproj, self, [mode for _ in self], num_workers=num_workers, name=f'deproj-{self.name}')
        path = '.'.join(self.path.split('.')[:-1]) + f'.conllu-deproj-{mode}'
        return CoNLL(nonp, path=path)
        
    def _proj(self, graph: CoNLL.Graph, mode: str) -> CoNLL.Graph:
        return graph.projectivize(mode)
    
    def _deproj(self, graph: CoNLL.Graph, mode: str) -> CoNLL.Graph:
        return graph.deprojectivize(mode)

        
class EnhancedCoNLL(CoNLL):
    """Representation of eh [Enhanced CoNLL-U](https://universaldependencies.org/v2/conll-u.html) format for Graph Parsing."""
            
    class Graph(SDP.Graph):
        def __init__(self, nodes: List[CoNLL.Node], arcs: List[Arc], annotations: List[Union[int, str]] = [], ID: Optional[int] = None):
            self.NODE = CoNLL.Node
            super(SDP.Graph, self).__init__(nodes, arcs, annotations, ID)
            
        @property
        def tags(self) -> List[str]:
            return [node.UPOS for node in self.nodes]
        
        @classmethod
        def from_raw(cls, lines: str) -> EnhancedCoNLL.Graph:
            """Parses a CoNLL block and extracts the enhanced dependencies.

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
                if len(_arcs) == 0:
                    node.DEPS = '_'
                else:
                    node.DEPS = '|'.join(f'{arc.HEAD}:{arc.REL}' for arc in _arcs)
            return cls(nodes, arcs, annotations)
        
        def rebuild(self, arcs: List[Arc]) -> EnhancedCoNLL.Graph:
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
    


class SentimentCoNLL(EnhancedCoNLL):
    class Graph(EnhancedCoNLL.Graph):
            
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
            matrix = np.zeros((len(nodes)+1, len(nodes)+1), dtype=bool)            
            for i, node in enumerate(nodes):
                if node.MISC == '_':
                    continue 
                values = node.MISC.split('|')
                _arcs = []
                for value in values:
                    head, *rel = value.split(':')
                    rel = ':'.join(rel)
                    if head.isdigit() and int(head) != node.ID and rel != '_': # avoid cycles
                        if not matrix[node.ID, int(head)]:
                            _arcs.append(Arc(int(head), node.ID, rel))
                            matrix[node.ID, int(head)] = True
                arcs += _arcs
                if len(_arcs) == 0:
                    node.MISC = '_'
                else:
                    node.MISC = '|'.join(f'{arc.HEAD}:{arc.REL}' for arc in _arcs)
                deps = node.DEPS 
                node.DEPS = node.MISC
                node.MISC = deps 
            return cls(nodes, arcs, annotations)
        

def propagate(arcs: List[Arc], _plane1: Set[Arc], _plane2: Set[Arc], arc: Arc, i: int) -> Tuple[Set[Arc], Set[Arc]]:
    """Propagation function from [Strzyz et al., (2020)](https://aclanthology.org/2020.coling-main.223/).

    Args:
        arcs (List[Arc]): List of input arcs.
        _plane1 (Set[Arc]): Set of arcs in plane 1.
        _plane2 (Set[Arc]): Set of arcs on plane 2.
        arc (Arc): New arc.
        i (int): Activated plane.

    Returns:
        Tuple[Set[Arc], Set[Arc]]: _description_
    """
    selected = _plane1 if i == 1 else _plane2
    no_selected = _plane1 if i == 2 else _plane2
    selected.add(arc)
    for other in arcs:
        if arc.cross(other) and other not in no_selected:
            _plane1, _plane2 = propagate(arcs, _plane1, _plane2, other, 3-i)
    return _plane1, _plane2
