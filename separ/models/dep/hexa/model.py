from typing import Optional, Tuple, List
from torch import nn
import torch 

from separ.model import Model 
from separ.utils import Config
from separ.modules import FFN 

class HexaTaggingDependencyModel(Model):
    
    def __init__(
        self, 
        enc_conf: Config, 
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        hexa_conf: Config,
        fence_conf: Config,
        con_conf: Config,
        rel_conf: Config,
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.hexa = FFN(self.hidden_size, hexa_conf.vocab_size)
        self.fence = FFN(self.hidden_size, fence_conf.vocab_size)
        self.con = FFN(self.hidden_size, con_conf.vocab_size)
        self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, hexa_conf, fence_conf, con_conf, rel_conf]
        self.hexa_conf = hexa_conf
        self.fence_conf = fence_conf
        self.con_conf = con_conf
        self.rel_conf = rel_conf
        
    def forward(
        self,
        words: torch.Tensor,
        feats: List[torch.Tensor],
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode(words, *feats)
        return self.hexa(embed[mask1]), self.fence(embed[mask1]), self.con(embed[mask1]), self.rel(embed[mask0])
    
    def loss(self, scores: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        return sum(self.criteria(score, pred) for score, pred in zip(scores, targets))
    
    def predict(
        self,
        words: torch.Tensor,
        feats: List[torch.Tensor],
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.postprocess(*self.forward(words, feats, mask0, mask1))
    
    def postprocess(
        self, 
        s_hexa: torch.Tensor, 
        s_fence: torch.Tensor, 
        s_con: torch.Tensor, 
        s_rel: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_hexa[:, self.hexa_conf.special_indices] = s_hexa.min()-1
        s_fence[:, self.fence_conf.special_indices] = s_fence.min()-1
        s_con[:, self.con_conf.special_indices] = s_con.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        return s_hexa.argmax(-1), s_fence.argmax(-1), s_con.argmax(-1), s_rel.argmax(-1)
    
    def control(
        self,
        words: torch.Tensor,
        feats: List[torch.Tensor],
        targets: List[torch.Tensor],
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.forward(words, feats, mask0, mask1)
        loss = self.loss(scores, targets)
        return loss, *self.postprocess(*scores)
