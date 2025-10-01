from __future__ import annotations
from typing import Iterable
from torch.optim import Optimizer, AdamW
from torch import nn 
import torch
from argparse import ArgumentParser

from separ.models.tag.tagger import Tagger 
from separ.models.tag.gat.model import GATModel
from separ.data import Sentence, Dataset, PretrainedTokenizer, TargetTokenizer
from separ.utils import ControlMetric, acc, TaggingMetric, Config, get_mask

class GaTagger(Tagger):
    NAME = 'gatag'
    MODEL = GATModel
    
    def OPTIMIZER(self, _, lr: float) -> tuple[Optimizer, Optimizer]:
        g_opt = AdamW([param for name, param in self.model.named_parameters() if not name.startswith('D_')], lr=lr)
        d_opt = AdamW([param for name, param in self.model.named_parameters() if name.startswith('D_')], lr=lr)
        return g_opt, d_opt 
    
    def optimize(self, loss: tuple[torch.Tensor, torch.Tensor], steps: int, max_norm: float, step: bool):
        (loss[0]/steps).backward(retain_graph=True)
        (loss[1]/steps).backward()

        if step:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
            self.optimizer[0].step()
            self.optimizer[1].step()
            
            self.optimizer[0].zero_grad()
            self.optimizer[1].zero_grad()
            
    def collate(self, sens: list[Sentence]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[Sentence]]:
        inputs = [tkz.batch_encode(sens, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(sens, pin=False) for tkz in self.target_tkzs]
        mask = get_mask(list(map(len, sens)))
        return inputs, [mask.clone() for _ in targets], targets, sens 
    
    def train_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        loss, scores = self.model(*inputs, masks, targets)
        targets = [target[mask] for target, mask in zip(targets, masks)]        
        return loss, ControlMetric(
            g_loss=loss[0].detach(), d_loss=loss[1].detach(), loss=sum(loss).detach(),
            **dict(zip(self.TARGET_FIELDS, map(acc, scores, targets)))
        )

    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        _
    ) -> tuple[ControlMetric, TaggingMetric, list[Sentence]]:
        loss, scores = self.model(*inputs, masks, targets)
        preds = self.model.postprocess(scores)
        targets = [target[mask] for target, mask in zip(targets, masks)]        
        return ControlMetric(g_loss=loss[0].detach(), d_loss=loss[1].detach()), self.METRIC(preds, targets), []
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        sens: list[Sentence]
    ) -> Iterable[Sentence]:
        preds = self.model.predict(*inputs, masks)
        return map(self._pred, sens, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)])
    
    @classmethod
    def build(
        cls,
        data: str | Dataset,
        word_conf: Config,
        dis_conf: Config,
        fields: list[str],
        device: int,
        **_
    ) -> GaTagger:
        if isinstance(data, str):
            data = cls.load_data(data)
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        dis_conf |= word_tkz.conf
        target_tkzs, target_confs = [], []
        for i, field in enumerate(fields):
            tkz = TargetTokenizer(f'TAG{i+1}' if len(fields) > 1 else 'TAG', field)
            tkz.train(data)
            target_tkzs.append(tkz)
            target_confs.append(tkz.conf)
        return cls([word_tkz], target_tkzs,  [word_conf, dis_conf, *target_confs], device)