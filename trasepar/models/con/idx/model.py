from typing import Optional, Tuple, List 
from torch import nn 
import torch 

from trasepar.model import Model 
from trasepar.modules import FFN 
from trasepar.utils import Config 

class IndexConstituencyModel(Model):
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        index_conf: Config,
        con_conf: Config,
        leaf_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.index_conf = index_conf
        self.con_conf = con_conf
        self.leaf_conf = leaf_conf
        self.index = FFN(self.hidden_size, index_conf.vocab_size)
        self.con = FFN(self.hidden_size, con_conf.vocab_size)
        self.leaf = FFN(self.hidden_size, leaf_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, index_conf, con_conf, leaf_conf]
        
    def forward(self, words: torch.Tensor, feats: List[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                s_index ~ [sum(seq_lens), n_indexes]: Flattened index scores per token.
                s_con ~ [sum(seq_lens), n_cons]: Flattened constituent scores per token.
                s_leaf ~ [sum(seq_lens), n_leafs]: Flattened terminal tag scores per token.
        """
        embed = self.encode(words, *feats)[mask]
        return self.index(embed), self.con(embed), self.leaf(embed)
    
    def loss(
        self, 
        s_index: torch.Tensor, s_con: torch.Tensor, s_leaf: torch.Tensor, 
        indexes: torch.Tensor, cons: torch.Tensor, leafs: torch.Tensor
    ) -> torch.Tensor:
        """Loss computation.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_con (torch.Tensor ~ [sum(seq_lens), n_cons]): Flattened constituent scores per token.
            s_leaf (torch.Tensor ~ [sum(seq_lens), n_leafs]): Flattened terminal tag scores per token.
            indexes (torch.Tensor ~ sum(seq_lens)): Gold index per token.
            cons (torch.Tensor ~ sum(seq_lens)): Gold constituents per token.
            leafs (torch.Tensor ~ sum(seq_lens)): Gold terminal leafs per token.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return self.criteria(s_index, indexes) + self.criteria(s_con, cons) + self.criteria(s_leaf, leafs)
    
    def predict(
        self,
        words: torch.Tensor,
        feats: List[torch.Tensor],
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Index, constituent and tag prediction.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            index_preds (torch.Tensor ~ sum(seq_lens)): Index predictions.
            con_preds (torch.Tensor ~ sum(seq_lens)): Constituent predictions.
            tag_preds (torch.Tensor ~ sum(seq_lens)): Terminal tag predictions.
        """
        s_index, s_con, s_leaf = self.forward(words, feats, mask)
        return self.postprocess(s_index, s_con, s_leaf)
        
    def postprocess(self, s_index: torch.Tensor, s_con: torch.Tensor, s_leaf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_index[:, self.index_conf.special_indices] = s_index.min()-1
        s_con[:, self.con_conf.special_indices] = s_con.min()-1
        s_leaf[:, self.leaf_conf.special_indices] = s_leaf.min()-1
        return s_index.argmax(-1), s_con.argmax(-1), s_leaf.argmax(-1)
        
    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor],
        indexes: torch.Tensor,
        cons: torch.Tensor,
        leaves: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (prediction + evaluation) step.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            indexes (torch.Tensor ~ sum(seq_lens)): Gold index per token.
            cons (torch.Tensor ~ sum(seq_lens)): Gold constituents per token.
            leafs (torch.Tensor ~ sum(seq_lens)): Gold terminal leafs per token.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        s_index, s_con, s_leaf = self.forward(words, feats, mask)
        return self.loss(s_index, s_con, s_leaf, indexes, cons, leaves), *self.postprocess(s_index, s_con, s_leaf)
        