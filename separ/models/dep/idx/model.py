from typing import Optional, List, Tuple
from torch import nn 
import torch 

from separ.model import Model 
from separ.modules import FFN 
from separ.utils import Config 


class IndexDependencyModel(Model):
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        index_conf: Config,
        rel_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.index_conf = index_conf 
        self.rel_conf = rel_conf 
        self.index = FFN(self.hidden_size, index_conf.vocab_size)
        self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, index_conf, rel_conf]
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Padded batch of input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Padded batch of input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: 
                s_index ~ [sum(seq_lens), n_indexes]: Flattened index scores per token.
                s_rel ~ [sum(seq_lens), n_rels]: Flattened relation scores per token.
        """
        embed = self.encode(words, *feats)[mask]
        return self.index(embed), self.rel(embed) 
    
    def loss(self, s_index: torch.Tensor, s_rel: torch.Tensor, indexes: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.
            indexes (torch.Tensor ~ sum(seq_lens)): Flattened batch of indexes.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened batch of rels.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return self.criteria(s_index, indexes) + self.criteria(s_rel, rels)
    
    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor],
        indexes: torch.Tensor, 
        rels: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (evaluation + prediction) step.

        Args:
            words (torch.Tensor): Padded batch of input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            indexes (torch.Tensor ~ sum(seq_lens)): Flattened batch of indexes.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened batch of rels.
            mask (torch.Tensor): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        scores = self.forward(words, feats, mask)
        loss = self.loss(*scores, indexes, rels)
        return loss, *self.postprocess(*scores)
    
    def postprocess(self, s_index: torch.Tensor, s_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores post-processing to suppress non-valid indices.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.
            
        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: Post-processed predictions.
        """
        s_index[:, self.index_conf.special_indices] = s_index.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        return s_index.argmax(-1), s_rel.argmax(-1)
    
    def predict(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index and REL prediction.

        Args:
            words (torch.Tensor): Padded batch of input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns: 
            index_preds (torch.Tensor ~ sum(seq_lens)]): Index predictions.
            rel_preds (torch.Tensor ~ sum(seq_lens)]): Relation predictions.
        """
        return self.postprocess(*self.forward(words, feats, mask))
    