from __future__ import annotations
from typing import Iterable
from argparse import ArgumentParser
import torch

from separ.models.tag.model import TagModel
from separ.utils import ControlMetric, TaggingMetric, acc, Config, get_mask
from separ.data import Dataset, Sentence, PretrainedTokenizer, CharacterTokenizer, InputTokenizer, TargetTokenizer, DATASET
from separ.parser import Parser


class Tagger(Parser):
    NAME = 'tag'
    MODEL = TagModel
    PARAMS = ['FIELDS']
    DATASET = DATASET
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('-f', '--fields', type=str, nargs='*', help='Fields to predict')
        return argparser
    
    @property
    def FIELDS(self) -> list[str]:
        return self.TARGET_FIELDS
    
    @property
    def METRIC(self) -> TaggingMetric:
        return TaggingMetric(self.target_tkzs)
    
    def collate(self, sens: list[Sentence]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[Sentence]]:
        inputs = [tkz.batch_encode(sens, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(sens, mode='cat', pin=False) for tkz in self.target_tkzs]
        mask = get_mask(list(map(len, sens)))
        return inputs, [mask.clone() for _ in targets], targets, sens 
    
    def train_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        return loss, ControlMetric(loss=loss.detach(), **dict(zip(self.TARGET_FIELDS, map(acc, scores, targets))))
            
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        _: list[Sentence]
    ) -> tuple[ControlMetric, TaggingMetric]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        return ControlMetric(loss=loss.detach()), self.METRIC(preds, targets)
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        sens: list[Sentence]
    ) -> Iterable[Sentence]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        preds = self.model.predict(scores)
        preds = [pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]
        return map(self._pred, sens, *preds)
            
    def _pred(self, sen: Sentence, *preds: list[torch.Tensor]) -> Sentence:
        for tkz, pred in zip(self.target_tkzs, preds):
            sen.rebuild(tkz.field, tkz.decode(pred))
        return sen 
            
    @classmethod
    def build(
        cls,
        data: str | Dataset,
        fields: list[str],
        enc_conf: Config,
        word_conf: Config,
        char_conf: Config | None = None,
        device: int = 0,
        **_
    ) -> Tagger:
        if isinstance(data, str):
            data = cls.load_data(data)
        
        if 'pretrained' in word_conf:
            word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        else:
            word_tkz = InputTokenizer('WORD', 'FORM')
            word_tkz.train(data)
        word_conf |= word_tkz.conf 
        input_tkzs = [word_tkz]
        if char_conf:
            char_tkz = CharacterTokenizer('CHAR', 'FORM')
            char_tkz.train(data)
            char_conf |= char_tkz.conf 
            input_tkzs.append(char_tkz)
        else:
            char_conf = None
            
        target_tkzs, target_confs = [], []
        for i, field in enumerate(fields):
            tkz = TargetTokenizer(f'TAG{i+1}' if len(fields) > 1 else 'TAG', field)
            tkz.train(data)
            target_tkzs.append(tkz)
            target_confs.append(tkz.conf)
        return Tagger(input_tkzs, target_tkzs, [enc_conf, word_conf, None, char_conf, *target_confs], device)
            