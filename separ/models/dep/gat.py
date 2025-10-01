from __future__ import annotations
import torch 

from separ.models.dep.sl import DependencySLParser
from separ.models.tag.gat.tagger import GaTagger
from separ.models.tag.gat.model import GATModel
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import avg, ControlMetric, acc, TaggingMetric, Config, bar, flatten

class DependencyGaTParser(DependencySLParser, GaTagger):
    MODEL = GATModel
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        trees: list[CoNLL.Tree]
    ) -> list[CoNLL.Tree]:
        preds = self.model.predict(*inputs, masks)
        preds, _ = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        return preds
    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, TaggingMetric, list[CoNLL.Tree]]:
        loss, scores = self.model(*inputs, masks, targets)
        preds = self.model.postprocess(scores)
        targets = [target[mask] for target, mask in zip(targets, masks)]    
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        return ControlMetric(
            g_loss=loss[0].detach(), d_loss=loss[1].detach(), loss=sum(loss).detach(),
            **dict(zip(self.TARGET_FIELDS, map(acc, scores, targets))), well_formed=avg(well_formed)*100
        ), self.TAG_METRIC(preds, targets), self.METRIC(pred_trees, trees), pred_trees
    
    
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        word_conf: Config, 
        dis_conf: Config,
        **_
    ) -> DependencyGaTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        dis_conf |= word_tkz.conf
            
        target_tkzs = cls.build_targets(cls.Labeler(**_), data)

        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=target_tkzs,
            model_confs=[word_conf, dis_conf] + [tkz.conf for tkz in target_tkzs], 
            **_
        )
