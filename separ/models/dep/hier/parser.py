from __future__ import annotations
from typing import List, Tuple, Union, Optional
from argparse import ArgumentParser
import torch 

from separ.models.dep.bracket.parser import BracketDependencyParser
from separ.models.dep.parser import DependencySLParser
from separ.data.struct import Bracket, Arc, Graph
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.utils import Config, flatten, bar

class HierarchicalBracketDependencyParser(BracketDependencyParser):
    NAME = 'dep-hier'
    PARAMS = ['variant']
    
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
        
        def __lt__(self, other: HierarchicalBracketDependencyParser.Bracket) -> str:
            return self.order < other.order
        
        def add_cross(self):
            self.p += 1
            
        @classmethod
        def from_string(cls, raw: str) -> List[HierarchicalBracketDependencyParser.Bracket]:
            brackets = []
            for c in raw:
                if c in cls.PRIORITY:
                    brackets.append(c)
                else:
                    brackets[-1] += c 
            return [cls(bracket[0], '^' in bracket, bracket.count('*'), None) for bracket in brackets]
    
    class Labeler(DependencySLParser.Labeler):
        
        def __init__(self, variant: str):
            self.variant = variant 
            
        def __repr__(self):
            return f'HierarchicalBracketLabeler(variant={self.variant})'
        
        def get_ropes(self, graph: Graph) -> List[Tuple[Arc, List[Arc]]]:
            """Gets the structural and auxiliar arcs of a graph.

            Args:
                graph (Graph): Input graph.

            Returns:
                List[Tuple[Arc, List[Arc]]]: List of optimal rope covers.
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
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            """Encodes the input graph.

            Args:
                graph (Graph): Input graph.

            Returns:
                Tuple[List[str], List[str]]: List of labels and dependency relations.
            """
            if self.variant == 'proj': # projective variant 
                labels = self.encode_proj(graph)
            elif self.variant == 'nonp': # non-projective variant
                labels = self.encode_nonp(graph)
            else: # pseudo-projectivity 
                graph = graph.projectivize(self.variant)
                labels = self.encode_proj(graph)
            return labels, [arc.REL for arc in graph.arcs]
        
        def encode_proj(self, graph: Graph) -> List[str]:
            ropes = self.get_ropes(graph)
            labels = [[] for _ in range(len(graph))]
            
            for structural, auxiliar in ropes:
                if structural.is_left():
                    labels[structural.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('<', structural=True, match=structural.HEAD))
                    labels[structural.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('\\', structural=True, match=structural.DEP))
                else:
                    labels[structural.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('>', structural=True, match=structural.HEAD))
                    if structural.HEAD != 0:
                        labels[structural.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('/', structural=True, match=structural.DEP))
                
                for arc in auxiliar:
                    if arc.left == structural.left:
                        labels[arc.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('>', structural=False, match=arc.HEAD))
                    else:
                        labels[arc.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('<', structural=False, match=arc.HEAD))
            return [''.join(map(repr, sorted(label))) for label in labels]
        
        def encode_nonp(self, graph: Graph) -> List[str]:
            ropes = self.get_ropes(graph)
            labels = [[] for _ in range(len(graph))]
            
            for structural, auxiliar in ropes:
                middle = [bracket for label in labels[structural.left:structural.right-1] for bracket in label]
                if structural.is_left():
                    for bracket in middle:
                        if bracket.is_left() and bracket.structural:
                            bracket.add_cross()
                    labels[structural.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('<', structural=True, p=0, match=structural.HEAD))
                    labels[structural.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('\\', structural=True, p=0, match=structural.DEP))
                else:
                    for bracket in middle:
                        if bracket.is_right() and bracket.structural:
                            bracket.add_cross()
                    labels[structural.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('>', structural=True, p=0, match=structural.HEAD))
                    if structural.HEAD != 0:
                        labels[structural.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('/', structural=True, p=0, match=structural.DEP))
                    
                for bracket in middle:
                    if (not bracket.structural) and bracket.match not in structural.range:
                        bracket.add_cross()
                
                for arc in auxiliar:
                    middle = [bracket for label in labels[arc.left:arc.right-1] for bracket in label]
                    if arc.HEAD == structural.left:
                        labels[arc.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('>', structural=False, p=0, match=arc.HEAD))
                    elif arc.HEAD == structural.right:
                        crosses = sum(bracket.is_closing() and bracket.structural for bracket in middle)
                        labels[arc.DEP-1].append(HierarchicalBracketDependencyParser.Bracket('<', structural=False, p=crosses, match=arc.HEAD))
                    elif arc.DEP == structural.left:
                        labels[arc.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('\\', structural=False, p=0, match=arc.DEP))
                    elif arc.DEP == structural.right:
                        crosses = sum(bracket.is_closing() and bracket.structural for bracket in middle)
                        labels[arc.HEAD-1].append(HierarchicalBracketDependencyParser.Bracket('/', structural=False, p=crosses, match=arc.DEP))
            return [''.join(map(repr, sorted(label))) for label in labels] 
        
        def decode(self, labels: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            stack = [(0, HierarchicalBracketDependencyParser.Bracket('/', structural=True, p=0, match=None))]
            adjacent = torch.zeros(len(labels)+1, len(labels)+1, dtype=torch.bool)
            well_formed = True
            for i, label in enumerate(labels):
                brackets = HierarchicalBracketDependencyParser.Bracket.from_string(label)
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
            return self.postprocess(adjacent, rels), len(stack) == 0 and well_formed and (adjacent.sum() == len(labels))
        
        def test(self, graph: CoNLL.Graph) -> bool:
            if self.variant != 'nonp':
                graph = graph.projectivize()
            return super().test(graph)
                
        
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        variant: str, 
        device: int
    ) -> HierarchicalBracketDependencyParser:
        super(BracketDependencyParser, self).__init__(input_tkzs, target_tkzs, model_confs, device)
        self.variant = variant
        self.lab = self.Labeler(variant)
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser = super().add_arguments(argparser)
        argparser.add_argument('-v', '--variant', type=str, default='proj', choices=['proj', 'head', 'path', 'head+path', 'nonp'], help='Projectivity selection')
        return argparser
    
    def _pred(self, graph: CoNLL.Graph, bracket_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[CoNLL.Graph, bool]:
        rec, well_formed = self.labeler.decode(self.BRACKET.decode(bracket_pred), self.REL.decode(rel_pred))
        pred = graph.rebuild(rec)
        if self.variant in ['head', 'path', 'head+path']:
            pred = pred.deprojectivize(self.variant)
        return pred, well_formed
    
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        variant: str = 'proj',
        device: int = 0,
        **_
    ) -> HierarchicalBracketDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)

            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
            
        bracket_tkz, rel_tkz = TargetTokenizer('BRACKET'), TargetTokenizer('REL')
        labeler = cls.Labeler(variant)
        brackets, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bracket_tkz.train(brackets)
        rel_tkz.train(rels)
        return cls(input_tkzs, [bracket_tkz, rel_tkz], [enc_conf, *in_confs, bracket_tkz.conf, rel_tkz.conf], variant, device)