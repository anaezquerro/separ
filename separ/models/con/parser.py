from argparse import ArgumentParser
import torch

from separ.data import PTB, InputTokenizer, TargetTokenizer
from separ.utils import ControlMetric, acc, avg, ConstituencyMetric, Config, TaggingMetric
from separ.models.tag import Tagger
from separ.models.tag.model import TagModel

class ConstituencySLParser(Tagger):
    MODEL = TagModel
    PARAMS = {'evalb'}
    DATASET = [PTB]
    
    def __init__(self, evalb: tuple[str, str, str] | None, **kwargs):
        super().__init__(**kwargs)
        self.evalb = evalb 
    
    @property
    def METRIC(self) -> ConstituencyMetric:
        if self.official:
            return ConstituencyMetric(*self.evalb)
        else:
            return ConstituencyMetric()
    
    @property
    def TAG_METRIC(self) -> TaggingMetric:
        return TaggingMetric(self.target_tkzs)
    
    @property
    def official(self) -> bool:
        return hasattr(self, 'evalb') and self.evalb is not None
    
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser = super().add_arguments(argparser)
        argparser.add_argument('--evalb', type=str, default=None, nargs=3, required=False, help='EVALB evaluation script')
        return argparser
    
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
    ) -> list[PTB.Tree]:
        preds, _ = zip(*super().pred_step(inputs, masks, trees))
        return preds 
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[PTB.Tree]
    ) -> tuple[ControlMetric, TaggingMetric, ConstituencyMetric, list[PTB.Tree]]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.TAG_METRIC(preds, targets), self.METRIC(pred_trees, trees), pred_trees
    
    @torch.no_grad()
    def analyze_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        trees: list[PTB.Tree]
    ) -> list[dict[str, float]]:
        preds = self.model.predict(self.model(inputs[0], inputs[1:], *masks))
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        preds = zip(*[pred.split(mask.sum(-1).tolist())  for pred, mask in zip(preds, masks)])
        targets = zip(*[target.split(mask.sum(-1).tolist())  for target, mask in zip(targets, masks)])
        scores = []
        for pred_tree, tree, pred, target, w in zip(pred_trees, trees, preds, targets, well_formed):
            score = {
                **dict(self.METRIC(pred_tree, tree).items()),
                'well_formed': w,
                **dict(zip(self.TARGET_FIELDS, map(float, map(acc, pred, target)))),
                'length': len(tree)
            }
            scores.append(score)
        return scores
    