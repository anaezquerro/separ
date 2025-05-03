from __future__ import annotations
import numpy as np 
import torch
from typing import List, Tuple, Iterator, Union, Optional

from separ.parser import Parser 
from separ.models.sdp.biaffine.model import BiaffineSemanticModel
from separ.utils import SemanticMetric, get_2d_mask, pad2D, Config, flatten, macro_fscore, ControlMetric
from separ.data import SDP, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, EnhancedCoNLL, Arc, Graph

class BiaffineSemanticParser(Parser):
    NAME = 'sdp-biaffine'
    MODEL = BiaffineSemanticModel
    METRIC = SemanticMetric
    DATASET = [SDP, EnhancedCoNLL]
    PARAMS = ['root_rel']
    
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        root_rel: Optional[str],
        device: int
    ) -> BiaffineSemanticParser:
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.root_rel = root_rel
    
    @property
    def METRIC(self) -> SemanticMetric:
        return SemanticMetric()
    
    def transform(self, graph: Graph) -> Graph:
        if not graph.transformed:
            graph.REL = torch.tensor(np.apply_along_axis(self.REL.encode, 1, graph.LABELED_ADJACENT))
            graph.transformed = True 
        return graph 
    
    def _pred(self, graph: Graph, arc_pred: torch.Tensor, rel_pred: torch.Tensor) -> Graph:
        arcs = [Arc(head, dep, self.root_rel if self.root_rel and head == 0 else self.REL.inv_vocab[rel_pred[dep, head].item()])\
            for dep, head in arc_pred.nonzero().tolist()]
        return graph.rebuild_from_arcs(arcs)
        
    def collate(self, graphs: List[Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Graph]]:
        inputs = [tkz.batch_encode(graphs) for tkz in self.input_tkzs]
        arcs = pad2D([graph.ADJACENT for graph in graphs]).to(torch.long)
        rels = pad2D([graph.REL for graph in graphs]).to(torch.long)
        masks = [get_2d_mask(list(map(len, graphs)), bos=True)]
        return inputs, masks, [arcs, rels], graphs
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ControlMetric]:
        s_arc, s_rel = self.model(inputs[0], inputs[1:])
        loss = self.model.loss(s_arc, s_rel, *targets, *masks)
        return loss, ControlMetric(loss=loss.detach(), ARC=macro_fscore(s_arc, targets[0]), REL=macro_fscore(s_rel, targets[1]))
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[Graph]
    ) -> Iterator[Graph]:
        arc_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, graphs, arc_preds.unbind(0), rel_preds.unbind(0))
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        targets: List[torch.Tensor],
        graphs: List[Graph]
    ) -> Tuple[ControlMetric, SemanticMetric]:
        loss, arc_preds, rel_preds = self.model.evaluate(inputs[0], inputs[1:], *targets, *masks)
        preds = list(map(self._pred, graphs, arc_preds.unbind(0), rel_preds.unbind(0)))
        return ControlMetric(loss=loss.detach()), SemanticMetric(preds, graphs)
    
    @classmethod
    def build(
        cls,
        data: Union[EnhancedCoNLL, SDP, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: int = 0,
        **_
    ) -> BiaffineSemanticModel:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=True)]
            in_confs = [word_conf | input_tkzs[0].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM', bos=True)]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'POS', bos=True))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM', bos=True))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                            
            for tkz in input_tkzs:
                tkz.train(data)
            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
                
        rel_tkz = TargetTokenizer('REL')
        rels = ([],[])
        for arc in flatten(graph.arcs for graph in data):
            rels[0 if arc.HEAD == 0 else 1].append(arc.REL)
        rel_tkz.train(rels[0] + rels[1])
        rel_conf = rel_tkz.conf
        if len(set(rels[0])) == 1:
            root_rel = rels[0][0]
            rel_conf.special_indices.append(rel_tkz.vocab[root_rel])
        else:
            root_rel = None
            
        return cls(input_tkzs, [rel_tkz], [enc_conf, *in_confs, rel_conf], root_rel, device)

