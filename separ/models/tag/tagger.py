from __future__ import annotations
from typing import List, Union, Tuple, Dict, Iterator
import torch 

from separ.data import Tokenizer, PretrainedTokenizer, CoNLL, PTB, SDP
from separ.parser import Parser
from separ.models.tag.model import TagModel
from separ.utils import TaggingMetric, create_mask, acc, Config, flatten

class Tagger(Parser):
    """Standard Tagging Model."""
    NAME = 'tag'
    MODEL = TagModel
    DATASET = [SDP, CoNLL, PTB]
    PARAMS = []
    TRANSFORM_ARGS = ['input_tkzs', 'target_tkzs']
    
    def __init__(
        self,
        model: TagModel,
        word_tkz: PretrainedTokenizer,
        target_tkzs: List[Tokenizer],
        device: Union[str, int]
    ):
        super().__init__(model, [word_tkz], target_tkzs, device)
        self.TRANSFORM_ARGS = [word_tkz, target_tkzs]
        self.METRIC = TaggingMetric(self.TARGET_FIELDS)
        
    @classmethod
    def transform(cls, sent: Union[CoNLL.Graph, PTB.Tree, SDP.Graph], WORD: PretrainedTokenizer, target_tkzs: List[Tokenizer]):
        if not sent._transformed:
            for tkz in [WORD] + target_tkzs:
                sent.__setattr__(tkz.name, tkz.encode(getattr(sent, tkz.field)).pin_memory())
            sent._transformed = True 
    
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
        s_tag = self.model(*inputs, *masks)
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
        tag_preds = self.model.predict(*inputs, *masks)
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
        s_tag = self.model(*inputs, *masks)
        loss = self.model.loss(s_tag, targets)
        tag_preds = self.model.predict(*inputs, *masks)
        return dict(loss=loss.item()), map(self._pred, sents, *[tag_pred.split(lens) for tag_pred in tag_preds])
    
    @classmethod
    def build(
        cls,
        data: Union[CoNLL, PTB, SDP, str],
        enc_conf: Config,
        word_conf: Config,
        fields: List[Tuple[str, str]],
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> Tagger:
        if isinstance(data, str):
            data = cls.load_data(data, num_workers)
            
        word_tkz = PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)
        word_conf = word_conf | word_tkz.conf

        target_tkzs = []
        for name, field in fields:
            tkz = Tokenizer(name, field)
            tkz.train(*flatten(getattr(sent, field) for sent in data))
            target_tkzs.append(tkz)
        
        model = cls.MODEL(enc_conf, word_conf, *[tkz.conf for tkz in target_tkzs]).to(device)
        return cls(model, word_tkz, target_tkzs, device)
