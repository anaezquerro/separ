from __future__ import annotations
import numpy as np 
import torch

from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc
from separ.utils import DependencyMetric, pad2D, fscore, get_mask, get_2d_mask, Config, acc, ControlMetric
from separ.models.dep.parser import DependencyParser
from separ.models.dep.biaffine.model import BiaffineDependencyModel

class BiaffineDependencyParser(DependencyParser):
    NAME = 'dep-biaffine'
    MODEL = BiaffineDependencyModel
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.REL = torch.tensor(np.apply_along_axis(self.REL.encode, 1, tree.LABELED_ADJACENT))
            tree.transformed = True 
        return tree
    
    def collate(self, batch: list[CoNLL.Tree]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[CoNLL.Tree]]:
        inputs = [tkz.batch_encode(batch, pin=False) for tkz in self.input_tkzs]
        arcs = pad2D([tree.ADJACENT for tree in batch]).to(torch.long)
        rels = pad2D([tree.REL for tree in batch]).to(torch.long)
        lens = list(map(len, batch))
        masks = [get_mask(lens, bos=True), get_2d_mask(lens, bos=True)]
        masks[0][:, 0] = False
        return inputs, masks, [arcs, rels], batch 
    
    def train_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor], 
    ) -> tuple[torch.Tensor, ControlMetric]:
        arcs, rels = targets
        s_arc, s_rel = self.model(inputs[0], inputs[1:])
        loss = self.model.loss(s_arc, s_rel, *targets, masks[0])
        arcs = arcs.to(torch.bool)
        return loss, ControlMetric(ARC=fscore(s_arc, arcs), REL=acc(s_rel[arcs], rels[arcs]), loss=loss.detach())
    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, DependencyMetric, list[CoNLL.Tree]]:
        arcs, rels, mask0, _ = *targets, *masks
        lens = mask0.sum(-1).tolist()
        loss, head_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        preds = list(map(self._pred, trees, head_preds.split(lens), rel_preds.split(lens)))
        return ControlMetric(loss=loss.detach(), ARC=acc(head_preds, arcs[mask0].argmax(-1)),  REL=acc(rel_preds, rels[arcs.to(torch.bool)])), \
            DependencyMetric(preds, trees), preds
            
    def _pred(self, tree: CoNLL.Tree, head_pred: torch.Tensor, rel_pred: torch.Tensor) -> CoNLL.Tree:
        arcs = [Arc(head, i+1, rel) for i, (head, rel) in enumerate(zip(head_pred.tolist(), self.REL.decode(rel_pred)))]
        return tree.rebuild_from_arcs(arcs)
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> list[CoNLL.Tree]:
        lens = masks[0].sum(-1).tolist()
        head_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, trees, head_preds.split(lens), rel_preds.split(lens))

    @classmethod
    def build(
        cls,
        data: CoNLL | str,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        device: int = 0,
        **_
    ) -> BiaffineDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        input_tkzs, in_confs = cls.build_inputs(data, word_conf, tag_conf, char_conf)
            
        rel_tkz = TargetTokenizer('REL')
        rel_tkz.train([arc.REL for tree in data for arc in tree.arcs])
        return cls(
            input_tkzs=input_tkzs, 
            target_tkzs=[rel_tkz],
            model_confs=[enc_conf, *in_confs, rel_tkz.conf], 
            device=device
        )

