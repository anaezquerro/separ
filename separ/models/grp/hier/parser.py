from __future__ import annotations
import os 

from separ.models.grp.sl import GraphSLParser
from separ.data import Graph, Arc
from separ.models.dep.hier.parser import HierarchicalDependencyParser


class HierarchicalGraphParser(GraphSLParser):
    """Hierarchical Bracketing Semantic parser."""
    
    NAME = 'grp-hier'
    DECOLLAPSE = True 
    
    class Labeler(GraphSLParser.Labeler, HierarchicalDependencyParser.Labeler):
        NAMES = ['HIER', 'REL']
        NAME = 'grp-hier'
        
        def __init__(self, **_):
            self.variant = 'nonp'
        
        def __repr__(self):
            return 'HierarchicalGraphLabeler()'
        
        
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            labels = self.encode_nonp(graph)
            return labels, self.encode_rels(graph.arcs, len(graph))

        def decode(self, labels: list[str], rels: list[str] | None = None) -> tuple[list[Arc], bool]:
            stack, arcs = [(0, HierarchicalDependencyParser.Bracket('/', structural=True, p=0, match=None))], []
            well_formed = True
            for i, label in enumerate(labels):
                brackets = HierarchicalDependencyParser.Bracket.from_string(label)
                for bracket in brackets:
                    added = False
                    if bracket.is_opening():
                        stack.append((i+1, bracket))
                        continue 
                    elif not bracket.structural: # closing auxiliar arc
                        # search for the p-th top opening structural arc 
                        for idx, s in stack[::-1]:
                            if s.structural and bracket.p == 0:
                                arcs.append(Arc(i+1, idx, None) if bracket.is_head() else Arc(idx, i+1, None))
                                added = True 
                                break 
                            elif s.structural:
                                bracket.p -= 1
                    else:
                        new = []
                        for j, (idx, s) in enumerate(stack[::-1]):
                            if s.structural and s.side == bracket.side and bracket.p == 0:
                                arcs.append(Arc(i+1, idx, None) if bracket.is_head() else Arc(idx, i+1, None))
                                added = True
                                break
                            elif s.structural and s.side == bracket.side:
                                bracket.p -= 1
                                new.append((idx, s))
                            elif s.structural:
                                new.append((idx, s))
                            else:
                                if s.p > 0:
                                    s.p -= 1
                                    new.append((idx, s))
                                else:
                                    arcs.append(Arc(idx, i+1, None) if s.is_head() else Arc(i+1, idx, None))
                                    added = True 
                        stack = stack[:-(j+1)] + new[::-1]
                    well_formed = well_formed and added 
            arcs = sorted(set(arcs))
            if rels is not None:
                arcs = self.decode_rels(arcs, rels)
            return arcs, len(stack) <= 1 and well_formed
                            