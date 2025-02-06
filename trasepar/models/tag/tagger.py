from __future__ import annotations
from typing import List, Union, Tuple, Dict, Iterator, Optional
import torch 

from trasepar.data import Tokenizer, PretrainedTokenizer, CoNLL, PTB, SDP, CharacterTokenizer, NER
from trasepar.parser import Parser 
from trasepar.models.tag.model import TagModel
from trasepar.structs import AbstractDataset
from trasepar.utils import TaggingMetric, create_mask, acc, Config, flatten


class Tagger(Parser):
    """Standard Tagging Model."""
    NAME = 'tag'
    MODEL = TagModel
    PARAMS = []
    TRANSFORM_ARGS = ['input_tkzs', 'target_tkzs']
    DATASET = [PTB, CoNLL, NER, SDP]
    
    def __init__(
        self,
        model: TagModel,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        device: Union[str, int]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.TRANSFORM_ARGS = [*input_tkzs, *target_tkzs]
        
        
    @property
    def METRIC(self):
        return TaggingMetric(self.TARGET_FIELDS)
        
    @classmethod
    def transform(cls, sent: Union[CoNLL.Graph, PTB.Tree, SDP.Graph], *tkzs: Tokenizer):
        if not sent._transformed:
            for tkz in tkzs:
                sent.__setattr__(tkz.name, tkz.encode(getattr(sent, tkz.field)).pin_memory())
            sent._transformed = True 
            
    @classmethod
    def add_arguments(cls, argparser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--fields', nargs='*', type=str, help='Fields to predict')
        return argparser 
    
    def collate(self, sents: List[Union[CoNLL.Graph, PTB.Tree, SDP.Graph]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in sents]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in sents]) for tkz in self.target_tkzs]
        masks = [create_mask(list(map(len, sents)))]
        return inputs, targets, masks, sents
    
    def _pred(self, sent: Union[CoNLL.Graph, SDP.Graph, PTB.Tree], *tag_preds: torch.Tensor) -> Union[CoNLL.Graph, SDP.Graph, PTB.Tree]:
        sent = sent.copy()
        for pred, tkz in zip(tag_preds, self.target_tkzs):
            sent.__setattr__(tkz.name, pred.to('cpu'))
        return sent 

    def train_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor], masks: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        s_tag = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_tag, targets)
        return loss, dict(zip(self.TARGET_FIELDS, map(acc, s_tag, targets)))
        
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        sents: List[Union[CoNLL.Graph, SDP.Graph, PTB.Tree]]
    ) -> Iterator[Union[CoNLL.Graph, SDP.Graph, PTB.Tree]]:
        lens = masks[0].sum(-1).tolist()
        tag_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        return map(self._pred, sents, *[tag_pred.split(lens) for tag_pred in tag_preds])
    
    @torch.no_grad()
    def control_step(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        masks: List[torch.Tensor],
        sents: List[Union[CoNLL.Graph, SDP.Graph, PTB.Tree]]
    ) -> Tuple[Dict[str, float], Iterator[Union[CoNLL.Graph, SDP.Graph, PTB.Tree]]]:
        lens = masks[0].sum(-1).tolist()
        s_tag = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_tag, targets)
        tag_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        control = dict(zip(self.TARGET_FIELDS, map(acc, s_tag, targets))) | dict(loss=loss.item())
        return control, map(self._pred, sents, *[tag_pred.split(lens) for tag_pred in tag_preds])
    
    @classmethod
    def build(
        cls,
        data: Union[CoNLL, PTB, SDP, str],
        enc_conf: Config,
        word_conf: Config,
        fields: List[str],
        char_conf: Optional[Config] = None,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> Tagger:
        if isinstance(data, str):
            data = cls.load_data(data, num_workers)
            
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            word_conf |= input_tkzs[0].conf
            in_confs = [word_conf, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
            
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.join(tkz.conf)

        target_tkzs = []
        names = ['TAG'] if len(fields) == 1 else [f'TAG{i}' for i in range(len(fields))]
        for name, field in zip(names, fields):
            tkz = Tokenizer(name, field)
            tkz.train(*flatten(getattr(sent, field) for sent in data))
            target_tkzs.append(tkz)
        
        model = cls.MODEL(enc_conf, *in_confs, *[tkz.conf for tkz in target_tkzs]).to(device)
        return cls(model, input_tkzs, target_tkzs, device)
