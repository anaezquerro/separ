from __future__ import annotations
from typing import List, Tuple, Dict, Union, Optional, Iterator 
from argparse import ArgumentParser
import torch

from trasepar.structs import Arc, forms_cycles, candidates_no_cycles
from trasepar.utils import flatten, Config, parallel, create_mask, DependencyMetric, acc, avg
from trasepar.data import CoNLL, AbstractTokenizer, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from trasepar.parser import Parser
from trasepar.models.dep.labeler import DependencyLabeler
from trasepar.models.dep.bracket.model import BracketDependencyModel

class BracketDependencyParser(Parser):
    """Bracket Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-bracket'
    MODEL = BracketDependencyModel
    PARAMS = ['k']
    METRIC = DependencyMetric
    DATASET = CoNLL
    
    class Labeler(DependencyLabeler):
        """Bracketing encoding for k-planar dependency trees"""
        DEFAULT = ''
        BRACKETS = ['<', '\\', '/', '>']
        
        def __init__(self, k: int = 2):
            self.k  = k
            
        def __repr__(self) -> str:
            return f'BracketDependencyLabeler(k={self.k})'
            
        def split_bracket(self, bracket: str) -> List[str]:
            stack = []
            for x in bracket:
                if x == '*':
                    stack[-1] += x 
                else:
                    stack.append(x)
            return stack 
            
        def preprocess(self, graph: CoNLL.Graph) -> Dict[int, List[Arc]]:
            planes =  {p: graph.planes[p] for p in range(min(len(graph.planes), self.k))}
            if len(planes) < self.k:
                planes.update({p: [] for p in range(len(planes), self.k)})
            return planes 
            
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
            bracket_count = [[{br: 0 for br in self.BRACKETS} for _ in range(len(graph))] for _ in range(self.k)]
            planes = self.preprocess(graph)
            # priority: <\/>
            for p, plane in planes.items():
                for arc in plane:
                    idep, ihead = arc.DEP - 1, arc.HEAD - 1
                    if arc.HEAD > arc.DEP: # left arc 
                        bracket_count[p][ihead]['\\'] += 1 
                        bracket_count[p][idep]['<'] += 1
                    else: # right arc 
                        if arc.HEAD > 0:
                            bracket_count[p][ihead]['/'] += 1
                        bracket_count[p][idep]['>'] += 1
            brackets = ['' for _ in range(len(graph))]
            for i in range(len(graph)):
                for label in self.BRACKETS:
                    for p in range(self.k):
                        if bracket_count[p][i][label] > 0:
                            brackets[i] += (label+'*'*p)*bracket_count[p][i][label] 
            return brackets, graph.DEPREL
        
        def decode(self, brackets: List[str], rels: List[str]) -> List[Arc]:
            # append the artificial in the right-arc stack of the first plane
            right = [[0] if p == 0 else [] for p in range(self.k)]
            left = [[] for _ in range(self.k)]
            arcs = []
            for idep, bracket in enumerate(brackets):
                dep = idep + 1
                if len(bracket) == 0:
                    continue 
                bracket = self.split_bracket(bracket)
                for b in bracket[::-1]:
                    plane = b.count('*')
                    if '>' in b:
                        if len(right[plane]) > 0:
                            arc = Arc(right[plane][-1], dep, None)
                            arcs.append(arc)
                            right[plane].pop(-1)
                    elif '/' in b:
                        right[plane].append(dep)
                    elif '\\' in b:
                        if len(left[plane]) > 0:
                            arc = Arc(dep, left[plane][-1], None)
                            arcs.append(arc)
                            left[plane].pop(-1)
                    elif '<' in b:
                        left[plane].append(dep)
            arcs = sorted(arcs)
            for arc, rel in zip(arcs, rels):
                arc.REL = rel
            return arcs  
        
        def decode_postprocess(self, brackets: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            # append the artificial in the right-arc stack of the first plane
            right = [[0] if p == 0 else [] for p in range(self.k)]
            left = [[] for _ in range(self.k)]
            adjacent = torch.zeros(len(brackets) + 1, len(brackets) + 1, dtype=torch.bool)
            arcs = []
            well_formed = True 
            for idep, bracket in enumerate(brackets):
                dep = idep + 1
                if len(bracket) == 0:
                    continue 
                bracket = self.split_bracket(bracket)
                for b in bracket[::-1]:
                    plane = b.count('*')
                    if '>' in b:
                        if len(right[plane]) > 0 and \
                            (not forms_cycles(adjacent, dep, right[plane][-1])) and \
                            (not adjacent[:, 0].any() or right[plane][-1] != 0) \
                            and (not adjacent[dep].any()):
                            arc = Arc(right[plane].pop(-1), dep, None)
                            arcs.append(arc)
                            adjacent[arc.DEP, arc.HEAD] = True 
                        else:
                            well_formed = False
                    elif '/' in b:
                        right[plane].append(dep)
                    elif '\\' in b:
                        if len(left[plane]) > 0 and \
                            (not forms_cycles(adjacent, left[plane][-1], dep)) and (not adjacent[left[plane][-1]].any()):
                            arc = Arc(dep, left[plane].pop(-1), None)
                            arcs.append(arc)
                            adjacent[arc.DEP, arc.HEAD] = True 
                        else:
                            well_formed = False
                    elif '<' in b:
                        left[plane].append(dep)
            well_formed = well_formed and len(arcs) == len(brackets) and len(flatten(left, right)) == 0
            arcs = sorted(self.postprocess(arcs, adjacent))
            for arc, rel in zip(arcs, rels):
                arc.REL = rel
            return arcs, well_formed

        def postprocess(self, arcs: List[Arc], adjacent: torch.Tensor) -> List[Arc]:
            no_assigned = set((adjacent.sum(-1) == 0).nonzero().flatten().tolist()[1:])
            for dep in no_assigned:
                head = candidates_no_cycles(adjacent, dep).pop(0)
                arcs.append(Arc(head, dep, None))
                adjacent[dep, head] = True
            return arcs 
        
        def test(self, graph: CoNLL.Graph) -> bool:
            k = self.k 
            if len(graph.planes) > self.k:
                self.k = len(graph.planes)
            brackets, rels = self.encode(graph)
            rec1 = graph.rebuild(self.decode(brackets, rels))
            rec2, well_formed = self.decode_postprocess(brackets, rels)
            rec2 = graph.rebuild(rec2)
            self.k = k
            return graph == rec1 == rec2 and well_formed
                    
    def __init__(
        self,
        model: BracketDependencyModel,
        input_tkzs: List[AbstractTokenizer],
        target_tkzs: List[AbstractTokenizer],
        k: int, 
        device: str
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.k = k
        self.labeler = self.Labeler(k)
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
         
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('-k', type=int, help='Number of planes')
        return argparser
    
    @classmethod
    def transform(
        cls, 
        graph: CoNLL.Graph, 
        input_tkzs: List[Tokenizer], 
        BRACKET: Tokenizer, 
        REL: Tokenizer, 
        labeler: BracketDependencyParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            brackets, rels = labeler.encode(graph)
            graph.BRACKET = BRACKET.encode(brackets).pin_memory()
            graph.REL = REL.encode(rels).pin_memory()
            graph._transformed = True 
            
    def collate(self, graphs: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.target_tkzs]
        mask = create_mask(list(map(len, graphs)))
        return inputs, targets, [mask], graphs
    
    def _pred(self, graph: CoNLL.Graph, bracket_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[CoNLL.Graph, bool]:
        rec, well_formed = self.labeler.decode_postprocess(self.BRACKET.decode(bracket_pred), self.REL.decode(rel_pred))
        return graph.rebuild(rec), well_formed
        
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        brackets, rels = targets
        s_bracket, s_rel = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_bracket, s_rel, brackets, rels)
        return loss, dict(BRACKET=acc(s_bracket, brackets), REL=acc(s_rel, rels))    
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Iterator[CoNLL.Graph]:
        lens = masks[0].sum(-1).tolist()
        bracket_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, graphs, bracket_preds.split(lens), rel_preds.split(lens)))
        return preds  
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        brackets, rels, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, bracket_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, mask)
        preds, well_formed = zip(*map(self._pred, graphs, bracket_preds.split(lens), rel_preds.split(lens)))
        control = dict(BRACKET=acc(bracket_preds, brackets), REL=acc(rel_preds, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
        return control, preds
        
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        k: int = 2,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> BracketDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'UPOS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
                
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.update(**tkz.conf())
            
        bracket_tkz, rel_tkz = Tokenizer('BRACKET'), Tokenizer('REL')
        labeler = cls.Labeler(k)
        if pretest:
            assert all(parallel(labeler.test, data, name=f'{cls.NAME}[pretest]'))
        brackets, rels = zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]'))
        bracket_tkz.train(*flatten(brackets))
        rel_tkz.train(*flatten(rels))
        
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        model = cls.MODEL(enc_conf, *in_confs, bracket_tkz.conf, rel_conf).to(device)
        return cls(model, input_tkzs, [bracket_tkz, rel_tkz], k, device)
