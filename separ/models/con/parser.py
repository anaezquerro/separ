from typing import Iterator 
import torch

from separ.data import PTB
from separ.utils import ControlMetric, acc, avg, ConstituencyMetric
from separ.models.tag import Tagger
from separ.models.tag.model import TagModel

class ConstituencySLParser(Tagger):
    DATASET = [PTB]
    MODEL = TagModel
    PARAMS = []
    
    @property 
    def METRIC(self) -> ConstituencyMetric:
        return ConstituencyMetric()
    
    def _pred(self, tree: PTB.Tree, *preds: list[torch.Tensor]) -> tuple[PTB.Tree, bool]:
        spans, well_formed = self.lab.decode(
            *[tkz.decode(pred) for pred, tkz in zip(preds, self.target_tkzs)]
        )
        rec = PTB.Tree.from_spans(tree.preterminals, spans).recover_unary()
        rec.ID = tree.ID
        return rec, well_formed
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        trees: list[PTB.Tree]
    ) -> Iterator[PTB.Tree]:
        preds, _ = zip(*super().pred_step(inputs, masks, trees))
        return preds 
    
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[PTB.Tree]
    ) -> tuple[ControlMetric, ConstituencyMetric]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.METRIC(pred_trees, trees)