from __future__ import annotations
from argparse import ArgumentParser
import torch

from separ.data import PTB, PretrainedTokenizer, InputTokenizer, CharacterTokenizer, TargetTokenizer
from separ.utils import ControlMetric, acc, avg, ConstituencyMetric, TaggingMetric, Config, flatten, bar
from separ.models.tag import Tagger
from separ.models.tag.model import TagModel

class ConstituencySLParser(Tagger):
    MODEL = TagModel
    DATASET = [PTB]
    
    def __init__(self, **_):
        super().__init__(**_)
        self.lab = self.Labeler(**_)
        
    @property
    def PARAMS(self) -> set[str]:
        return {'evalb'} | self.Labeler.PARAMS
    
    @property
    def METRIC(self) -> ConstituencyMetric:
        if self.official:
            return ConstituencyMetric(*self.evalb)
        else:
            return ConstituencyMetric()
        
    def transform(self, tree: PTB.Tree) -> PTB.Tree:
        if not tree.transformed:
            for name, labels in zip(self.lab.NAMES, self.lab.encode(tree)):
                setattr(tree, name, labels)
            tree.transformed = True 
        return tree 
    
    @property
    def TAG_METRIC(self) -> TaggingMetric:
        return TaggingMetric(self.target_tkzs)
    
    @property
    def official(self) -> bool:
        return hasattr(self, 'evalb') and self.evalb is not None
    
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
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
    
    @classmethod
    def build_inputs(
        cls, 
        data: PTB, 
        word_conf: Config,
        char_conf: Config | None = None,
    ) -> tuple[list[InputTokenizer], list[Config]]:
        if hasattr(word_conf, 'pretrained'):
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            word_tkz = InputTokenizer('WORD', 'FORM')
            word_tkz.train(data)
            word_conf.update(word_tkz.conf)
            input_tkzs = [word_tkz]
            
            if char_conf is not None:
                char_tkz = CharacterTokenizer('CHAR', 'FORM')
                char_tkz.train(data)
                char_conf.update(char_tkz.conf)
                input_tkzs.append(char_tkz)
                
            in_confs = [word_conf, None, char_conf]
            
        return input_tkzs, in_confs
    
    @classmethod
    def build_targets(
        cls,
        lab: ConstituencySLParser.Labeler, 
        data: PTB
    ) -> list[TargetTokenizer]:
        target_tkzs = [TargetTokenizer(name) for name in lab.NAMES]
        labels = zip(*bar(map(lab.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]'))
        for tkz, _labels in zip(target_tkzs, map(flatten, labels)):
            tkz.train(_labels)
        return target_tkzs
  
    @classmethod
    def build(
        cls, 
        data: str | PTB,
        enc_conf: Config,
        word_conf: Config,
        char_conf: Config | None = None,
        **_
    ) -> ConstituencySLParser:
        if isinstance(data, str):
            data = PTB.from_file(data)
        input_tkzs, in_confs = cls.build_inputs(data, word_conf, char_conf)
        target_tkzs = cls.build_targets(cls.Labeler(**_), data)
        return cls(
            input_tkzs=input_tkzs, 
            target_tkzs=target_tkzs, 
            model_confs=[enc_conf, *in_confs] + [tkz.conf for tkz in target_tkzs],
            **_
        )
            