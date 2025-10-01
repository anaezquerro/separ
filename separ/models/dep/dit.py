
from __future__ import annotations
import torch 

from separ.models.tag.dit.tagger import DiTagger
from separ.models.tag.dit.model import DiTModel
from separ.models.dep.sl import DependencySLParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import DependencyMetric, ControlMetric, acc, avg, TaggingMetric, Config, flatten, bar

class DependencyDiTParser(DependencySLParser, DiTagger):
    MODEL = DiTModel
    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, TaggingMetric, DependencyMetric, list[CoNLL.Tree]]:
        loss = self.model.loss(*self.model(*inputs, masks, targets))
        preds = self.model.predict(*inputs, masks)
        targets = [target[mask] for target, mask in zip(targets, masks)]
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.TAG_METRIC(preds, targets), self.METRIC(pred_trees, trees), pred_trees
    
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        dec_conf: Config,
        **_
    ) -> DependencyDiTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
            
        target_tkzs = cls.build_targets(cls.Labeler(**_), data)

        return cls(
            input_tkzs=[word_tkz],
            target_tkzs=target_tkzs, 
            model_confs=[enc_conf, word_conf, dec_conf] + [tkz.conf for tkz in target_tkzs], 
            **_
        )
        
        
