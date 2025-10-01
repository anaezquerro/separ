from __future__ import annotations
from argparse import ArgumentParser
import torch 

from separ.models.dep.bracket.parser import BracketDependencyParser
from separ.models.dep.sl import DependencySLParser
from separ.data.struct import Bracket, Arc, Graph
from separ.data import deprojectivize

class HierarchicalDependencyParser(BracketDependencyParser):
    NAME = 'dep-hier'
    
    class Bracket(Bracket):
        def __init__(self, symbol: str, structural: bool, p: int = 0, match: int = 0):
            super().__init__(symbol, p)
            self.structural = structural 
            self.match = match
            
        @property
        def order(self) -> int:
            if self.is_closing():
                return (0, -self.match, self.structural)
            else:
                return (1, -self.match, self.structural)
                        
        def __repr__(self) -> str:
            return f'{self.symbol}' + ('^'*self.structural) + '*'*self.p
        
        def __lt__(self, other: HierarchicalDependencyParser.Bracket) -> str:
            return self.order < other.order
        
        def add_cross(self):
            self.p += 1
            
        @classmethod
        def from_string(cls, raw: str) -> list[HierarchicalDependencyParser.Bracket]:
            brackets = []
            for c in raw:
                if c in cls.PRIORITY:
                    brackets.append(c)
                else:
                    brackets[-1] += c 
            return [cls(bracket[0], '^' in bracket, bracket.count('*'), None) for bracket in brackets]
    
    class Labeler(DependencySLParser.Labeler):
        PARAMS = {'variant'}
        NAMES = ['BRACKET', 'REL']
        
        def __init__(self, variant: str, **_):
            self.variant = variant 
            
        def __repr__(self):
            return f'HierarchicalBracketLabeler(variant={self.variant})'
        
        @property
        def NAMES(self) -> list[str]:
            if self.variant == 'nonp':
                return ['HIER(np)', 'DEPREL']
            elif self.variant == 'proj':
                return ['HIER', 'DEPREL']
            else:
                abbr = ''.join(t[0] for t in self.proj.split('+'))
                return ['HIER', f'DEPREL({abbr})']
            
        @property
        def NAME(self) -> str:
            abbr = ''.join(t[0] for t in self.proj.split('+'))
            return f'dep-hier({abbr})'
                
        def get_ropes(self, graph: Graph) -> list[tuple[Arc, list[Arc]]]:
            """Gets the structural and auxiliar arcs of a graph.

            Args:
                graph (Graph): Input graph.

            Returns:
                list[tuple[Arc, list[Arc]]]: List of optimal rope covers.
            """
            arcs = sorted(graph.arcs, key=lambda arc: (arc.left, -len(arc)))
            ropes = []
            while len(arcs) > 0:
                structural, auxiliar = arcs.pop(0), []
                remain = []
                for arc in arcs:
                    if arc in structural and (arc.left == structural.left or arc.right == structural.right):
                        auxiliar.append(arc)
                    else:
                        remain.append(arc)
                arcs = remain
                ropes.append((structural, auxiliar))
            return ropes
        
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            """Encodes the input graph.

            Args:
                graph (Graph): Input graph.

            Returns:
                tuple[list[str], list[str]]: List of labels and dependency relations.
            """
            if self.variant == 'proj': # projective variant 
                labels = self.encode_proj(graph)
            elif self.variant == 'nonp': # non-projective variant
                labels = self.encode_nonp(graph)
            else: # pseudo-projectivity 
                graph = graph.projectivize(self.variant)
                labels = self.encode_proj(graph)
            return labels, [arc.REL for arc in graph.arcs]
        
        def encode_proj(self, graph: Graph) -> list[str]:
            ropes = self.get_ropes(graph)
            labels = [[] for _ in range(len(graph))]
            
            for structural, auxiliar in ropes:
                if structural.is_left():
                    labels[structural.DEP-1].append(HierarchicalDependencyParser.Bracket('<', structural=True, match=structural.HEAD))
                    labels[structural.HEAD-1].append(HierarchicalDependencyParser.Bracket('\\', structural=True, match=structural.DEP))
                else:
                    labels[structural.DEP-1].append(HierarchicalDependencyParser.Bracket('>', structural=True, match=structural.HEAD))
                    if structural.HEAD != 0:
                        labels[structural.HEAD-1].append(HierarchicalDependencyParser.Bracket('/', structural=True, match=structural.DEP))
                
                for arc in auxiliar:
                    if arc.left == structural.left:
                        labels[arc.DEP-1].append(HierarchicalDependencyParser.Bracket('>', structural=False, match=arc.HEAD))
                    else:
                        labels[arc.DEP-1].append(HierarchicalDependencyParser.Bracket('<', structural=False, match=arc.HEAD))
            return [''.join(map(repr, sorted(label))) for label in labels]
        
        def encode_nonp(self, graph: Graph) -> list[str]:
            ropes = self.get_ropes(graph)
            labels = [[] for _ in range(len(graph))]
            
            for structural, auxiliar in ropes:
                middle = [bracket for label in labels[structural.left:structural.right-1] for bracket in label]
                if structural.is_left():
                    for bracket in middle:
                        if bracket.is_left() and bracket.structural:
                            bracket.add_cross()
                    labels[structural.DEP-1].append(HierarchicalDependencyParser.Bracket('<', structural=True, p=0, match=structural.HEAD))
                    labels[structural.HEAD-1].append(HierarchicalDependencyParser.Bracket('\\', structural=True, p=0, match=structural.DEP))
                else:
                    for bracket in middle:
                        if bracket.is_right() and bracket.structural:
                            bracket.add_cross()
                    labels[structural.DEP-1].append(HierarchicalDependencyParser.Bracket('>', structural=True, p=0, match=structural.HEAD))
                    if structural.HEAD != 0:
                        labels[structural.HEAD-1].append(HierarchicalDependencyParser.Bracket('/', structural=True, p=0, match=structural.DEP))
                    
                for bracket in middle:
                    if (not bracket.structural) and bracket.match not in structural.range:
                        bracket.add_cross()
                
                for arc in auxiliar:
                    middle = [bracket for label in labels[arc.left:arc.right-1] for bracket in label]
                    if arc.HEAD == structural.left:
                        labels[arc.DEP-1].append(HierarchicalDependencyParser.Bracket('>', structural=False, p=0, match=arc.HEAD))
                    elif arc.HEAD == structural.right:
                        crosses = sum(bracket.is_closing() and bracket.structural for bracket in middle)
                        labels[arc.DEP-1].append(HierarchicalDependencyParser.Bracket('<', structural=False, p=crosses, match=arc.HEAD))
                    elif arc.DEP == structural.left:
                        labels[arc.HEAD-1].append(HierarchicalDependencyParser.Bracket('\\', structural=False, p=0, match=arc.DEP))
                    elif arc.DEP == structural.right:
                        crosses = sum(bracket.is_closing() and bracket.structural for bracket in middle)
                        labels[arc.HEAD-1].append(HierarchicalDependencyParser.Bracket('/', structural=False, p=crosses, match=arc.DEP))
            return [''.join(map(repr, sorted(label))) for label in labels] 
        
        def decode(self, labels: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            stack = [(0, HierarchicalDependencyParser.Bracket('/', structural=True, p=0, match=None))]
            adjacent = torch.zeros(len(labels)+1, len(labels)+1, dtype=torch.bool)
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
                            head, dep = (i+1, idx) if bracket.is_head() else (idx, i+1)
                            if s.structural and bracket.p == 0 and self.is_valid(adjacent, dep, head):
                                adjacent[dep, head] = True 
                                added = True 
                                break 
                            elif s.structural:
                                bracket.p -= 1
                    else:
                        new = []
                        for j, (idx, s) in enumerate(stack[::-1]):
                            head, dep = (i+1, idx) if bracket.is_head() else (idx, i+1)
                            if s.structural and s.side == bracket.side and bracket.p == 0 and self.is_valid(adjacent, dep, head):
                                adjacent[dep, head] = True 
                                added = True
                                break
                            elif s.structural and s.side == bracket.side:
                                bracket.p -= 1
                                new.append((idx, s))
                            elif s.structural:
                                new.append((idx, s))
                            else:
                                head, dep = (idx, i+1) if s.is_head() else (i+1, idx)
                                if s.p > 0:
                                    s.p -= 1
                                    new.append((idx, s))
                                elif self.is_valid(adjacent, dep, head):
                                    adjacent[dep, head] = True 
                                    added = True 
                        stack = stack[:-(j+1)] + new[::-1]
                    well_formed = well_formed and added 
            arcs = self.postprocess(adjacent, rels)
            if self.variant in ['head', 'path', 'head+path']:
                arcs = deprojectivize(arcs, self.variant)
            return arcs, len(stack) == 0 and well_formed and (adjacent.sum() == len(labels))
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser.add_argument('-v', '--variant', type=str, default='proj', choices=['proj', 'head', 'path', 'head+path', 'nonp'], help='Projectivity selection')
        return argparser