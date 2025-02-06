from __future__ import annotations
from typing import List, Union, Tuple, Dict, Iterator, Optional
import numpy as np 
import torch, os

from trasepar.data import CoNLL, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from trasepar.utils import DependencyMetric, pad2D, fscore, create_2d_mask, Config, flatten, create_mask, acc
from trasepar.structs import MST, Arc
from trasepar.parser import Parser
from trasepar.models.dep.biaffine.model import BiaffineDependencyModel

class BiaffineDependencyParser(Parser):
    NAME = 'dep-biaffine'
    MODEL = BiaffineDependencyModel
    DATASET = CoNLL 
    METRIC = DependencyMetric
    PARAMS = []
    
    def __init__(
        self, 
        model: BiaffineDependencyModel,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        device: Union[str, int]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.TRANSFORM_ARGS = [input_tkzs, self.REL]
    
    @classmethod
    def transform(cls, graph: CoNLL.Graph, input_tkzs: List[Tokenizer], REL: Tokenizer):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)))
            graph.REL = torch.tensor(np.apply_along_axis(REL.encode, 1, graph.LABELED_ADJACENT)).pin_memory()
            graph._transformed = True 
    
    def collate(self, batch: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        assert all(graph._transformed for graph in batch), 'Dataset is not transformed'
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        arcs = pad2D([graph.ADJACENT for graph in batch]).to(torch.long)
        rels = pad2D([graph.REL for graph in batch]).to(torch.long)
        lens = list(map(len, batch))
        masks = [create_mask(lens, bos=True), create_2d_mask(lens, bos=True)]
        masks[0][:, 0] = False
        return inputs, [arcs, rels], masks, batch 
    
    def train_step(
        self, 
        inputs: Tuple[torch.Tensor], 
        targets: Tuple[torch.Tensor], 
        masks: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        arcs, rels = targets
        s_arc, s_rel = self.model(inputs[0], inputs[1:])
        loss = self.model.loss(s_arc, s_rel, *targets, masks[0])
        arcs = arcs.to(torch.bool)
        return loss, dict(ARC=fscore(s_arc, arcs), REL=acc(s_rel[arcs], rels[arcs]))
    
    @torch.no_grad()
    def pred_step(self, inputs: List[torch.Tensor], masks: List[torch.Tensor], graphs: List[CoNLL.Graph]) -> Iterator[CoNLL.Graph]:
        lens = masks[0].sum(-1).tolist()
        head_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, graphs, head_preds.split(lens), rel_preds.split(lens))

    def _pred(self, graph: CoNLL.Graph, head_pred: torch.Tensor, rel_pred: torch.Tensor) -> CoNLL.Graph:
        arcs = [Arc(head, i+1, rel if head != 0 else 'root') for i, (head, rel) in enumerate(zip(head_pred.tolist(), self.REL.decode(rel_pred)))]
        return graph.rebuild(arcs)
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor],
        masks: List[torch.Tensor],
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        arcs, rels, mask0, _ = *targets, *masks
        lens = mask0.sum(-1).tolist()
        loss, head_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        return dict(loss=loss.item(), ARC=acc(head_preds, arcs[mask0].argmax(-1)),  REL=acc(rel_preds, rels[arcs.to(torch.bool)])), \
            map(self._pred, graphs, head_preds.split(lens), rel_preds.split(lens))
    
    @classmethod
    def build(
        cls,
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: str = 'cuda:0',
        num_workers: int = os.cpu_count(),
        **_
    ) -> BiaffineDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data, num_workers)
            
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained, bos=True)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM', bos_token='<bos>')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'UPOS', bos_token='<bos>'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM', bos_token='<bos>'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
                
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.update(**tkz.conf())
                
        rel_tkz = Tokenizer('REL')
        rel_tkz.train(*[arc.REL for graph in data for arc in graph.arcs])
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        model = cls.MODEL(enc_conf, *in_confs, rel_conf).to(device)
        return cls(model, input_tkzs, [rel_tkz], device)

