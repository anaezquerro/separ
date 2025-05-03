from typing import Tuple, List, Iterator 
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
    
    def _pred(self, tree: PTB.Tree, *preds: List[torch.Tensor]) -> Tuple[PTB.Tree, bool]:
        spans, well_formed = self.lab.decode(
            *[tkz.decode(pred) for pred, tkz in zip(preds, self.target_tkzs)]
        )
        rec = PTB.Tree.from_spans(tree.preterminals, spans).recover_unary()
        rec.ID = tree.ID
        return rec, well_formed
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        trees: List[PTB.Tree]
    ) -> Iterator[PTB.Tree]:
        words, *feats, mask = *inputs, *masks
        lens = mask.sum(-1).tolist()
        preds = self.model.predict(self.model(words, feats, mask))
        preds, _ = zip(*map(self._pred, trees, *[pred.split(lens) for pred in preds]))
        return preds  
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        trees: List[PTB.Tree]
    ) -> Tuple[ControlMetric, ConstituencyMetric]:
        words, *feats, mask = *inputs, *masks
        lens = mask.sum(-1).tolist()
        scores = self.model(words, feats, mask)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(lens) for pred in preds]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.METRIC(pred_trees, trees)