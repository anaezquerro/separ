from __future__ import annotations
from typing import List, Union, Optional, Dict, Iterable 
import re 
import numpy as np 

from separ.utils.logger import bar
from separ.utils.fn import flatten
from separ.data.struct import Graph, Dataset, Arc, has_cycles

class CoNLL(Dataset):
    SEP = '\n\n'
    EXTENSION = 'conllu'

    class Tree(Graph):
        class Node(Graph.Node):
            FIELDS = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
            
            def __init__(self, *args):
                for field, arg in zip(self.FIELDS, args):
                    if field in ['ID', 'HEAD']:
                        arg = int(arg)
                    self.__setattr__(field, arg)

        def __init__(
            self,
            nodes: List[CoNLL.Tree.Node], 
            arcs: List[Arc],
            ID: Optional[int] = None,
            annotations: Optional[List[Union[str, int]]] = None
        ):
            """
            Initialization of a CoNLL tree.
            
            Args:
                nodes (List[CoNLL.Tree.Node]): Nodes of the tree (tokens).
                ID (Optional[int]): Number of the tree in the global dataset.
                annotations (Optional[List[Union[str, int]]]): List of annotations or node position.
            """
            super().__init__(nodes, arcs, ID, annotations)
            assert len(arcs) == len(nodes), 'Number of arcs and nodes must match'
            assert not has_cycles(self.ADJACENT), 'A dependency tree does not have cycles'
            
                
        def rebuild_from_arcs(self, new_arcs: List[Arc]) -> CoNLL.Tree:
            """Rebuilds a dependency tree from a new list of arcs.

            Args:
                new_arcs (List[Arc]): New arcs.

            Returns:
                CoNLL.Tree: New dependency tree.
            """
            new_nodes = []
            for node, arc in zip(self.nodes, sorted(new_arcs)):
                new_nodes.append(node.copy())
                new_nodes[-1].HEAD = arc.HEAD
                new_nodes[-1].DEPREL = arc.REL
            return CoNLL.Tree(new_nodes, new_arcs, ID=self.ID, annotations=self.annotations)
                
        def copy(self) -> CoNLL.Tree:
            tree = super().copy()
            tree.annotations = self.annotations.copy()
            return tree 
        
        @classmethod
        def from_raw(cls, block: str, ID: Optional[int] = None) -> CoNLL.Tree:
            lines = block.strip().split(cls.SEP)
            annotations, nodes, arcs = [], [], []
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.split()[0].isdigit():
                    annotations.append(len(nodes))
                    node = cls.Node.from_raw(line)
                    arcs.append(Arc(HEAD=node.HEAD, DEP=node.ID, REL=node.DEPREL))
                    nodes.append(node)
                else:
                    annotations.append(line)
            return cls(nodes, arcs, ID, annotations)
        
        @property
        def heads(self) -> np.ndarray:
            return np.array(self.HEAD)
        
        @property
        def rels(self) -> np.ndarray:
            return np.array(self.DEPREL, dtype=np.object_)
        
        @property
        def planes(self) -> Dict[int, List[Arc]]:
            """Breadth-first plane assignment (starting from node 0 to next-level dependents). 
            
            Returns:
                Dict[int, List[Arc]]: Plane assignment.
            """
            if self._planes is None:
                # store the planes 
                self._planes = {0: []}
                for arc in bfs(self.arcs):
                    added = False 
                    for p in self._planes.keys():
                        if not any(map(arc.cross, self._planes[p])):
                            self._planes[p].append(arc)
                            added = True 
                            break 
                    if not added:
                        self._planes[len(self._planes)] = [arc]
            return self._planes
        
        @property
        def relaxed_planes(self) -> Dict[int, List[Arc]]:
            """Breadth-first relaxed plane assignment. Relaxed planes allow crossing arcs in different directions.

            Returns:
                Dict[int, List[Arc]]: Relaxed plane assignment.
            """
            if self._relaxed_planes is None:
                self._relaxed_planes = {0: []}
                for arc in bfs(self.arcs):
                    added = False
                    for p in self._relaxed_planes.keys():
                        if not any(map(arc.relaxed_cross, self._relaxed_planes[p])):
                            self._relaxed_planes[p].append(arc)
                            added = True 
                            break 
                    if not added:
                        self._relaxed_planes[len(self._relaxed_planes)] = [arc]
            return self._relaxed_planes
        
        
        def is_projective(self) -> bool:
            return len(self.planes) == 1
       
        def projectivize(self, mode: str = 'head') -> CoNLL.Tree:
            """Pseudo-projective algorithm from [Nivre and Nilsson, (2005)](https://aclanthology.org/P05-1013/).

            Args:
                mode (str): Mode to encode the lifting operations.

            Returns:
                CoNLL.Tree: Projective dependency tree.
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
                proj = proj.rebuild_from_arcs(proj.arcs)
            return proj 
        
        def deprojectivize(self, mode: str = 'head') -> CoNLL.Tree:
            """Pseudo-deprojective algorithm from [Nivre and Nilsson, (2005)](https://aclanthology.org/P05-1013/).

            Args:
                mode (str): Mode to decode the lifting operations.

            Returns:
                CoNLL.Tree: Non-projective recovered dependency tree.
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
            return nonp.rebuild_from_arcs(nonp.arcs)
        
        def planarize(self, k: int) -> CoNLL.Tree:
            if len(self.planes) <= k:
                return self.copy()
            fixed = [arc.copy() for p in range(min(k, len(self.planes))) for arc in self.planes[p]]
            crosses = [arc.copy() for p in range(k, len(self.planes)) for arc in self.planes[p]]
            for arc in crosses:
                arc.HEAD = self.ADJACENT[arc.HEAD].int().argmax().item()
                planar = self.rebuild_from_arcs(fixed + crosses).planarize(k)
                return planar 
                
        def relaxed_planarize(self, k: int) -> CoNLL.Tree:
            if len(self.relaxed_planes) <= k:
                return self.copy()
            fixed = [arc.copy() for p in range(min(k, len(self.relaxed_planes))) for arc in self.relaxed_planes[p]]
            crosses = [arc.copy() for p in range(k, len(self.relaxed_planes)) for arc in self.relaxed_planes[p]]
            for arc in crosses:
                arc.HEAD = self.ADJACENT[arc.HEAD].int().argmax().item()
                planar = self.rebuild_from_arcs(fixed + crosses).planarize(k)
                return planar 
        
    @classmethod
    def from_file(cls, path: str) -> CoNLL:
        blocks = re.split('\n{2,}', open(path, 'r').read().strip())
        trees = list(bar(map(CoNLL.Tree.from_raw, blocks, range(len(blocks))), total=len(blocks), leave=False, desc=path))
        return cls(trees, path)
    
    
def bfs(arcs: List[Arc]) -> Iterable[Arc]:
    """Breadth-first search of the arcs of a tree."""
    queue = [0]
    while len(queue) > 0:
        head = queue.pop(0)
        deps = [arc for arc in arcs if arc.HEAD == head]
        for dep in deps:
            yield dep
            queue.append(dep.DEP)
            
        
