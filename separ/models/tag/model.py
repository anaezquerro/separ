from typing import List 
from torch import nn 
import torch 

from separ.model import Model
from separ.modules import FFN
from separ.utils import Config

class TagModel(Model):
    def __init__(self, enc_conf: Config, word_conf: Config, *target_confs: List[Config]):
        super().__init__(enc_conf, word_conf, None, None)
        self.target_confs = target_confs 
        for conf in target_confs:
            self.__setattr__(conf.name.lower(), FFN(enc_conf.hidden_size, conf.vocab_size))
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf] + target_confs
        
    def forward(self, words: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        embed = self.encode(words)[mask]
        return [tag(embed) for tag in self.tag]
    
    def loss(self, s_tag: List[torch.Tensor], tags: List[torch.Tensor]) -> torch.Tensor:
        return sum(map(self.criteria, s_tag, tags))
    
    def predict(self, words: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        embed = self.encode(words)[mask]
        preds = []
        for conf in self.target_confs:
            s_tag = self.__getattr__(conf.name)(embed)
            s_tag[:, conf.special_indices] = s_tag.min() - 1
            preds.append(s_tag.argmax(-1))
        return preds 