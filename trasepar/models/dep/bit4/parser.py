from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Tuple, Union, Optional, Dict, Iterator
import torch 

from trasepar.data import CoNLL, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from trasepar.utils import flatten, Config, parallel, DependencyMetric, create_mask, avg, acc
from trasepar.structs import Arc, forms_cycles, candidates_no_cycles
from trasepar.models.dep.bit4.model import Bit4DependencyModel
from trasepar.parser import Parser
from trasepar.models.dep.labeler import DependencyLabeler

class Bit4DependencyParser(Parser):
    """4-bit Dependency Parser from [Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)."""
    NAME = 'dep-bit4'
    MODEL = Bit4DependencyModel
    METRIC = DependencyMetric
    DATASET = CoNLL
    PARAMS = ['proj'] 
    
    class Labeler(DependencyLabeler):
        """4-bit encoding for projective dependency trees.
        
        - b0: word is a left dependant (0) or a right dependant (1) in the plane.
        - b1: word is the farthest dependant in the plane.
        - b2: word has left dependants in the plane.
        - b3: word has right dependants in the plane.
        """
        NUM_BITS = 4
        DEFAULT = '0000'
        
        def __init__(self, proj: Optional[str] = None):
            self.proj = proj
        
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
            if self.proj:
                graph = graph.projectivize(self.proj)
            bits = [[False for _ in range(self.NUM_BITS)] for _ in range(len(graph))]
            for idep, head in enumerate(graph.HEAD):
                dep = idep+1
                bits[idep][0] = head < dep
                bits[idep][1] = (graph.ADJACENT[:, head].nonzero().max().item() == dep) if head < dep \
                    else (graph.ADJACENT[:, head].nonzero().min().item() == dep)
                bits[idep][2] = graph.ADJACENT[:dep, dep].any().item()
                bits[idep][3] = graph.ADJACENT[dep:, dep].any().item()
            return [''.join(map(str, map(int, label))) for label in bits], list(graph.DEPREL)
        
        def decode(self, bits: List[str], rels: List[str]) -> List[Arc]:
            left, right = [], [0]
            arcs = []
            for idep, (label, rel) in enumerate(zip(bits, rels)):
                b0, b1, b2, b3 = map(bool, map(int, label))
                if b0: # right dependant
                    arcs.append(Arc(right[-1], idep+1, rel))
                    if b1: # farthest dependant 
                        right.pop(-1)
                if b2:
                    while not left[-1][-1]:
                        arcs.append(Arc(idep+1, *left.pop(-1)[:-1]))
                    arcs.append(Arc(idep+1, *left.pop(-1)[:-1]))
                if not b0:
                    left.append((idep+1, rel, b1))
                if b3:
                    right.append(idep+1)
            return sorted(arcs)
        
        def decode_postprocess(self, bits: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            left, right = [], [0]
            arcs, well_formed = [], True 
            adjacent = torch.zeros(len(bits)+1, len(bits)+1, dtype=torch.bool)
            for idep, label in enumerate(bits):
                b0, b1, b2, b3 = map(bool, map(int, label))
                if b0 and len(right) > 0: # right dependant
                    if not forms_cycles(adjacent, idep+1, right[-1]) and (not adjacent[idep+1].any()) \
                        and (not adjacent[:, 0].any() or right[-1] != 0):
                        arcs.append(Arc(right[-1], idep+1, None))
                        adjacent[idep+1, right[-1]] = True
                        if b1: # farthest dependant 
                            right.pop(-1)
                    else:
                        well_formed = False
                if b2:
                    last = False 
                    while len(left) > 0 and not last:
                        if not forms_cycles(adjacent, left[-1][0], idep+1) and (not adjacent[left[-1][0]].any()):
                            dep, last = left.pop(-1)
                            adjacent[dep, idep+1] = True
                            arcs.append(Arc(idep+1, dep, None))
                        else:
                            well_formed = False
                            break
                if not b0:
                    left.append((idep+1, b1))
                if b3:
                    right.append(idep+1)
            well_formed = well_formed and (len(right + left) == 0) and len(arcs) == len(bits)
            arcs = sorted(self.postprocess(arcs, adjacent))
            for arc, rel in zip(arcs, rels):
                arc.REL = rel if arc.HEAD != 0 else 'root'
            return arcs, well_formed
        
        def postprocess(self, arcs: List[Arc], adjacent: torch.Tensor) -> List[Arc]:
            no_assigned = sorted(set((adjacent.sum(-1) == 0).nonzero().flatten().tolist()[1:]))
            for dep in no_assigned:
                head = candidates_no_cycles(adjacent, dep).pop(0)
                arcs.append(Arc(head, dep, None))
                adjacent[dep, head] = True
            return arcs 
                
        def test(self, graph: CoNLL.Graph) -> bool:
            if len(graph.planes) == 1:
                bits, rels = self.encode(graph)
                rec1 = graph.rebuild(self.decode(bits, rels))
                rec2, well_formed = self.decode_postprocess(bits, rels)
                rec2 = graph.rebuild(rec2)
                return graph == rec1 == rec2 and well_formed
            else:
                planar = graph.planarize(1)
                bits, rels = self.encode(planar)
                rec1 = planar.rebuild(self.decode(bits, rels))
                rec2, well_formed = self.decode_postprocess(bits, rels)
                rec2 = planar.rebuild(rec2)
                return planar == rec1 == rec2 and well_formed
    
    def __init__(
        self,
        model: Bit4DependencyModel,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        proj: Optional[str],
        device: Union[str, int]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.labeler = self.Labeler(proj)
        self.proj = proj 
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--proj', default=None, type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 
    
        
    @classmethod
    def transform(cls, graph: CoNLL.Graph, input_tkzs: List[Tokenizer], BIT: Tokenizer, REL: Tokenizer,  labeler: Bit4DependencyParser.Labeler):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            bits, rels = labeler.encode(graph)
            graph.BIT = BIT.encode(bits).pin_memory()
            graph.REL = REL.encode(rels).pin_memory()
            graph._transformed = True 
            
    def collate(self, graphs: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.target_tkzs]
        masks = [create_mask(list(map(len, graphs)))]
        return inputs, targets, masks, graphs
    
    def _pred(
        self,
        graph: CoNLL.Graph, 
        bit_pred: torch.Tensor, 
        rel_pred: torch.Tensor
    ) -> Tuple[CoNLL.Graph, bool]:
        rec, well_formed = self.labeler.decode_postprocess(self.BIT.decode(bit_pred), self.REL.decode(rel_pred))
        pred = graph.rebuild(rec)
        if self.proj:
            pred = pred.deprojectivize(self.proj)
        return graph.rebuild(rec), well_formed

    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bits, rels = targets
        s_bit, s_rel = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_bit, s_rel, bits, rels)
        return loss, dict(BIT=acc(s_bit, bits), REL=acc(s_rel, rels))
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Iterator[CoNLL.Graph]:
        lens = masks[0].sum(-1).tolist()
        bit_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, graphs, bit_preds.split(lens), rel_preds.split(lens)))
        return preds  
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        bits, rels, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, bit_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, mask)
        preds, well_formed = zip(*map(self._pred, graphs, bit_preds.split(lens), rel_preds.split(lens)))
        control = dict(BIT=acc(bit_preds, bits), REL=acc(rel_preds, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
        return control, preds
        
    @classmethod
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        proj: Optional[str] = None,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> Bit4DependencyParser:
        if isinstance(data, str):
            data = cls.load_data(data, num_workers)
            
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            word_conf |= input_tkzs[0].conf
            in_confs = [word_conf, None, None]
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
        
        bit_tkz, rel_tkz = Tokenizer('BIT'), Tokenizer('REL')
        labeler = cls.Labeler(proj)
        if pretest:
            assert all(parallel(labeler.test, data, name=f'{cls.NAME}[pretest]'))
        bits, rels = map(flatten, zip(*parallel(labeler.encode, data, name=f'{cls.NAME}[encode]')))
        bit_tkz.train(*bits)
        rel_tkz.train(*rels)
        
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        model = cls.MODEL(enc_conf, *in_confs, bit_tkz.conf, rel_conf).to(device)
        return cls(model, input_tkzs, [bit_tkz, rel_tkz], proj, device)
