from __future__ import annotations
from typing import List, Union, Tuple, Dict, Iterator, Optional
import numpy as np 
import torch, os

from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc
from separ.utils import DependencyMetric, pad2D, fscore, get_mask, get_2d_mask, Config, flatten, acc, ControlMetric
from separ.parser import Parser
from separ.models.dep.biaffine.model import BiaffineDependencyModel

class BiaffineDependencyParser(Parser):
    NAME = 'dep-biaffine'
    MODEL = BiaffineDependencyModel
    DATASET = [CoNLL]
    
    def __init__(
        self, 
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        device: int 
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
    
    @property 
    def METRIC(self) -> DependencyMetric:
        return DependencyMetric()
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.REL = torch.tensor(np.apply_along_axis(self.REL.encode, 1, tree.LABELED_ADJACENT))
            tree.transformed = True 
        return tree
    
    def collate(self, batch: List[CoNLL.Tree]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Tree]]:
        inputs = [tkz.batch_encode(batch, pin=False) for tkz in self.input_tkzs]
        arcs = pad2D([tree.ADJACENT for tree in batch]).to(torch.long)
        rels = pad2D([tree.REL for tree in batch]).to(torch.long)
        lens = list(map(len, batch))
        masks = [get_mask(lens, bos=True), get_2d_mask(lens, bos=True)]
        masks[0][:, 0] = False
        return inputs, masks, [arcs, rels], batch 
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        targets: List[torch.Tensor], 
    ) -> Tuple[torch.Tensor, ControlMetric]:
        arcs, rels = targets
        s_arc, s_rel = self.model(inputs[0], inputs[1:])
        loss = self.model.loss(s_arc, s_rel, *targets, masks[0])
        arcs = arcs.to(torch.bool)
        return loss, ControlMetric(ARC=fscore(s_arc, arcs), REL=acc(s_rel[arcs], rels[arcs]), loss=loss.detach())
    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: List[torch.Tensor],
        masks: List[torch.Tensor],
        targets: List[torch.Tensor],
        trees: List[CoNLL.Tree]
    ) -> Tuple[ControlMetric, DependencyMetric]:
        arcs, rels, mask0, _ = *targets, *masks
        lens = mask0.sum(-1).tolist()
        loss, head_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        preds = list(map(self._pred, trees, head_preds.split(lens), rel_preds.split(lens)))
        return ControlMetric(loss=loss.detach(), ARC=acc(head_preds, arcs[mask0].argmax(-1)),  REL=acc(rel_preds, rels[arcs.to(torch.bool)])), \
            DependencyMetric(preds, trees)
            
    def _pred(self, tree: CoNLL.Tree, head_pred: torch.Tensor, rel_pred: torch.Tensor) -> CoNLL.Tree:
        arcs = [Arc(head, i+1, rel) for i, (head, rel) in enumerate(zip(head_pred.tolist(), self.REL.decode(rel_pred)))]
        return tree.rebuild_from_arcs(arcs)
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        trees: List[CoNLL.Tree]
    ) -> Iterator[CoNLL.Tree]:
        lens = masks[0].sum(-1).tolist()
        head_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, trees, head_preds.split(lens), rel_preds.split(lens))

    @classmethod
    def build(
        cls,
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: int = 0,
        **_
    ) -> BiaffineDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=True)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM', bos=True)]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS', bos=True))
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
        rel_tkz.train([arc.REL for tree in data for arc in tree.arcs])
        return cls(input_tkzs, [rel_tkz], [enc_conf, *in_confs, rel_tkz.conf], device)

