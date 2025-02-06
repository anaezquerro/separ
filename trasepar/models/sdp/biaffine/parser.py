from __future__ import annotations
import numpy as np 
import torch, os 
from typing import List, Tuple, Dict, Union, Optional


from trasepar.parser import Parser 
from trasepar.models.sdp.biaffine.model import BiaffineSemanticModel
from trasepar.utils import SemanticMetric, create_2d_mask, pad2D, Config, flatten, macro_fscore
from trasepar.data import SDP, Tokenizer, PretrainedTokenizer, CharacterTokenizer, EnhancedCoNLL
from trasepar.structs import Arc 

class BiaffineSemanticParser(Parser):
    NAME = 'sdp-biaffine'
    MODEL = BiaffineSemanticModel
    METRIC = SemanticMetric
    DATASET = [SDP, EnhancedCoNLL]
    PARAMS = ['root_rel']
    
    def __init__(
        self,
        model: BiaffineSemanticModel,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        root_rel: Optional[str],
        device: Union[str, int]
    ) -> BiaffineSemanticParser:
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.root_rel = root_rel
        self.TRANSFORM_ARGS = [input_tkzs, self.REL]
    
    @classmethod
    def transform(cls, graph: SDP.Graph, input_tkzs: List[Tokenizer], REL: Tokenizer):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)))
            graph.REL = torch.tensor(np.apply_along_axis(REL.encode, 1, graph.LABELED_ADJACENT)).pin_memory()
            graph._transformed = True 
            
    def _pred(self, graph: SDP.Graph, arc_pred: torch.Tensor, rel_pred: torch.Tensor) -> SDP.Graph:
        arcs = [Arc(head, dep, self.root_rel if self.root_rel and head == 0 else self.REL.inv_vocab[rel_pred[dep, head].item()])\
            for dep, head in arc_pred.nonzero().tolist()]
        return graph.rebuild(arcs)
        
    def collate(self, batch: List[SDP.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[SDP.Graph]]:
        assert all(graph._transformed for graph in batch), 'Dataset is not transformed'
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        arcs = pad2D([graph.ADJACENT for graph in batch]).to(torch.long)
        rels = pad2D([graph.REL for graph in batch]).to(torch.long)
        masks = [create_2d_mask(list(map(len, batch)), bos=True)]
        return inputs, [arcs, rels], masks, batch 
    
    def train_step(self, inputs: Tuple[torch.Tensor], targets: Tuple[torch.Tensor], masks: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        s_arc, s_rel = self.model(inputs[0], inputs[1:])
        loss = self.model.loss(s_arc, s_rel, *targets, *masks)
        return loss, dict(ARC=macro_fscore(s_arc, targets[0]), REL=macro_fscore(s_rel, targets[1]))
    
    @torch.no_grad()
    def pred_step(self, inputs: List[torch.Tensor], masks: List[torch.Tensor], graphs: List[SDP.Graph]) -> List[SDP.Graph]:
        arc_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, graphs, arc_preds.unbind(0), rel_preds.unbind(0))
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor],
        masks: List[torch.Tensor],
        graphs: List[SDP.Graph]
    ) -> Tuple[Dict[str, float], List[SDP.Graph]]:
        loss, arc_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        return dict(loss=loss.item()), map(self._pred, graphs, arc_preds.unbind(0), rel_preds.unbind(0))
    
    @torch.no_grad()
    def ref_step(self, targets: List[torch.Tensor], masks: List[torch.Tensor], graphs: List[SDP.Graph]) -> List[SDP.Graph]:
        arc_preds, rel_preds = targets
        return map(self._pred, graphs, arc_preds.unbind(0), rel_preds.unbind(0)) 

    @classmethod
    def build(
        cls,
        data: Union[SDP, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: str = 'cuda:0',
        num_workers: int = os.cpu_count(),
        **_
    ) -> BiaffineSemanticModel:
        if isinstance(data, str):
            data = cls.load_data(data, num_workers)
            
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained, bos=True)]
            in_confs = [word_conf | input_tkzs[0].conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM', bos_token='<bos>')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'POS', bos_token='<bos>'))
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
                conf.update(vocab_size=len(tkz), pad_index=tkz.pad_index)
                
        rel_tkz = Tokenizer('REL')
        rels = ([],[])
        for arc in flatten(graph.arcs for graph in data):
            rels[0 if arc.HEAD == 0 else 1].append(arc.REL)
        rel_tkz.train(*rels[0], *rels[1])
        rel_conf = rel_tkz.conf
        if len(set(rels[0])) == 1:
            root_rel = rels[0][0]
            rel_conf.special_indices.append(rel_tkz.vocab[root_rel])
        else:
            root_rel = None
            
        model = cls.MODEL(enc_conf, *in_confs, rel_conf).to(device)
        return cls(model, input_tkzs, [rel_tkz], root_rel, device)


