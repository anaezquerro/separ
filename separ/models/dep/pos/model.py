from typing import Optional, List, Tuple 
from torch import nn 
import torch 

from separ.model import Model
from separ.utils import Config
from separ.modules import FFN

class PoSDependencyModel(Model):
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        char_conf: Optional[Config],
        index_conf: Config,
        rel_conf: Config,
        tag_conf: Config
    ):
        super().__init__(enc_conf, word_conf, None, char_conf)
        self.index_conf, self.rel_conf, self.tag_conf = index_conf, rel_conf, tag_conf
        self.confs = [enc_conf, word_conf, char_conf, index_conf, rel_conf, tag_conf]
        
        self.index = FFN(enc_conf.hidden_size, index_conf.vocab_size)
        self.tag = FFN(enc_conf.hidden_size, tag_conf.vocab_size) 
        self.rel = FFN(enc_conf.hidden_size, rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                s_rel ~ [sum(seq_lens), n_rels]: Flattened REL scores per token.
                s_tag ~ [sum(seq_lens), n_tags]: Flattened tag scores per token.
        """
        embed = self.encode(words, *feats)[mask]
        return self.index(embed), self.rel(embed), self.tag(embed)
    
    def loss(
        self, 
        s_index: torch.Tensor, 
        s_rel: torch.Tensor, 
        s_tag: torch.Tensor,
        indexes: torch.Tensor, 
        rels: torch.Tensor, 
        tags: torch.Tensor,
    ) -> torch.Tensor:
        """Loss computation.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened relation scores per token.
            s_tag (torch.Tensor ~ [sum(seq_lens), n_tags]): Flattened tag scores per token.
            indexes (torch.Tensor ~ sum(seq_lens)): Flattened indexes per token.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened rel per token.
            tags (torch.Tensor ~ sum(seq_lens)): Flattened tags per token.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return self.criteria(s_index, indexes) + self.criteria(s_rel, rels) + self.criteria(s_tag, tags)
        
    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        indexes: torch.Tensor, 
        rels: torch.Tensor, 
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (evaluation + prediction) step.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            indexes (torch.Tensor ~ sum(seq_lens)): Flattened indexes per token.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened rel per token.
            tags (torch.Tensor ~ sum(seq_lens)): Flattened tags per token.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        s_index, s_rel, s_tag = self.forward(words, feats, mask)
        loss = self.loss(s_index, s_rel, s_tag, indexes, rels, tags)
        return loss, *self.postprocess(s_index, s_rel, s_tag)
    
    def postprocess(self, s_index: torch.Tensor, s_rel: torch.Tensor, s_tag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scores post-processing to suppress non-valid indices.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.
            s_tag (torch.Tensor ~ [sum(seq_lens), n_tags]): Flattened tag scores per token.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Post-processed predictions.
        """
        s_index[:, self.index_conf.special_indices] = s_index.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        s_tag[:, self.tag_conf.special_indices] = s_tag.min()-1
        return s_index.argmax(-1), s_rel.argmax(-1), s_tag.argmax(-1)
    
    def predict(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            tag_preds (torch.Tensor ~ sum(seq_lens)): Tag predictions.
        """
        return self.postprocess(*self.forward(words, feats, mask))