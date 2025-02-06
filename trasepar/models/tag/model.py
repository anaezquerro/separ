from typing import List
from torch import nn 
import torch 

from trasepar.model import Model 
from trasepar.modules import FFN 
from trasepar.utils import Config 

class TagModel(Model):
    def __init__(
        self, 
        enc_conf: Config, 
        word_conf: Config, 
        char_conf: Config,
        *target_confs: Config
    ):
        super().__init__(enc_conf, word_conf, None, char_conf)
        self.target_confs = list(target_confs)
        for conf in target_confs:
            self.__setattr__(conf.name.lower(), FFN(self.hidden_size, conf.vocab_size))
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, char_conf] + self.target_confs
        
    def forward(self, words: torch.Tensor, feats: List[torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
        embed = self.encode(words, *feats)[mask]
        return [self.__getattr__(conf.name)(embed) for conf in self.target_confs]
    
    def loss(self, s_tag: List[torch.Tensor], tags: List[torch.Tensor]) -> torch.Tensor:
        return sum(map(self.criteria, s_tag, tags))
    
    def predict(self, words: torch.Tensor, feats: List[torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
        embed = self.encode(words, *feats)[mask]
        preds = []
        for conf in self.target_confs:
            s_tag = self.__getattr__(conf.name)(embed)
            s_tag[:, conf.special_indices] = s_tag.min() - 1
            preds.append(s_tag.argmax(-1))
        return preds 