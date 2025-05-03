from typing import Optional, List, Tuple
from torch import nn
import torch 

from separ.model import Model
from separ.modules import FFN
from separ.utils import Config

class TetraTaggingConstituencyModel(Model):
    
    def __init__(
        self, 
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        tetra_conf: Config,
        fence_conf: Config,
        con_conf: Config,
        leaf_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.tetra_conf = tetra_conf
        self.fence_conf = fence_conf
        self.con_conf = con_conf
        self.leaf_conf = leaf_conf
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, 
                      tetra_conf, fence_conf, con_conf, leaf_conf]
        
        self.tetra = FFN(self.hidden_size, tetra_conf.vocab_size)
        self.fence = FFN(self.hidden_size, fence_conf.vocab_size)
        self.con = FFN(self.hidden_size, con_conf.vocab_size)
        self.leaf = FFN(self.hidden_size, leaf_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode(words, *feats)
        return self.tetra(embed[mask1]), self.fence(embed[mask1]), self.con(embed[mask1]), self.leaf(embed[mask0])
    
    def loss(
        self,
        s_tetra: torch.Tensor, s_fence: torch.Tensor, s_con: torch.Tensor, s_leaf: torch.Tensor,
        tetras: torch.Tensor, fences: torch.Tensor, cons: torch.Tensor, leaves: torch.Tensor
    ) -> torch.Tensor:
        return self.criteria(s_tetra, tetras) + self.criteria(s_fence, fences) + \
            self.criteria(s_con, cons) + self.criteria(s_leaf, leaves)
            
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
        s_tetra: torch.Tensor,
        s_fence: torch.Tensor, 
        s_con: torch.Tensor,
        s_leaf: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_tetra[:, self.tetra_conf.special_indices] = s_tetra.min()-1
        s_fence[:, self.fence_conf.special_indices] = s_fence.min()-1
        s_con[:, self.con_conf.special_indices] = s_con.min()-1
        s_leaf[:, self.leaf_conf.special_indices] = s_leaf.min()-1
        return s_tetra.argmax(-1), s_fence.argmax(-1), s_con.argmax(-1), s_leaf.argmax(-1)
    
    def control(
        self, 
        words: torch.Tensor, feats: List[torch.Tensor],
        tetras: torch.Tensor, fences: torch.Tensor, cons: torch.Tensor, leaves: torch.Tensor,
        mask0: torch.Tensor, mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.forward(words, feats, mask0, mask1)
        return self.loss(*scores, tetras, fences, cons, leaves), *self.postprocess(*scores)
