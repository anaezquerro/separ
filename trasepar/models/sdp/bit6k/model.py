from typing import Optional, List, Tuple 
from torch import nn 
import torch 

from trasepar.modules import FFN 
from trasepar.utils import Config 
from trasepar.models.sdp.idx.model import IndexSemanticModel
from trasepar.model import Model

class Bit6kSemanticModel(IndexSemanticModel):
    # inherits rel_score, rel_pred, rel_post
    
    def __init__(
        self, 
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        bit_conf: Config,
        rel_conf: Config
    ):
        Model.__init__(self, enc_conf, word_conf, tag_conf, char_conf)
        self.bit_conf = bit_conf
        self.rel_conf = rel_conf
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, bit_conf, rel_conf]
        
        self.bit = FFN(self.hidden_size, bit_conf.vocab_size)
        self.head = FFN(self.hidden_size, self.hidden_size//2)
        self.dep = FFN(self.hidden_size, self.hidden_size//2)
        self.rel = FFN(self.hidden_size, rel_conf.vocab_size)

        self.criteria = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        matrices: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched and padded adjacent matrices. 
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: 
                s_bit ~ [sum(seq_lens), n_bits]: Flattened label scores per token.
                s_rel ~ [sum(matrices), n_rels]: Flattened REL scores per gold arc.
        """
        embed = self.encode(words, *feats)
        return self.bit(embed)[mask], self.rel_score(embed, matrices, int(mask.sum()))
    
    def loss(self, s_bit: torch.Tensor, s_rel: torch.Tensor, bits: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_bit (torch.Tensor ~ [sum(seq_lens), n_bits]): Flattened bit scores per token.
            s_rel (torch.Tensor, [sum(matrices), n_rels]): Flattened relation scores per gold arc.
            bits (torch.Tensor ~ sum(seq_lens)): Gold bits per token.
            rels (torch.Tensor ~ sum(matrices)): Gold relations per gold arc.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        loss = self.criteria(s_bit, bits)
        if len(rels) > 0:
            loss += self.criteria(s_rel, rels.to(torch.long))
        return loss
    
    def control(
        self, 
        words: torch.Tensor,
        feats: List[torch.Tensor],
        bits: torch.Tensor, 
        rels: torch.Tensor, 
        matrices: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (evaluation + prediction) step.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            bits (torch.Tensor ~ sum(seq_lens)): Gold bits per token.
            rels (torch.Tensor ~ sum(matrices)): Gold relations per token.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched and padded adjacent matrices. 
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            loss (torch.Tensor): Cross-entropy loss.
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            bit_preds (torch.Tensor ~ max(seq_lens)): Flattened label predictions.
            s_rel (torch.Tensor ~ [max(seq_lens), n_rels]): Flattened relation scores.
        """
        embed = self.encode(words, *feats)
        s_bit = self.bit(embed)[mask]
        s_rel = self.rel_score(embed, matrices, int(mask.sum()))
        loss = self.loss(s_bit, s_rel, bits, rels)
        return loss, embed, self.bit_post(s_bit), s_rel
    
    def bit_post(self, s_bit: torch.Tensor) -> torch.Tensor:
        """Bit scores post-processing and prediction.

        Args:
            s_bit (torch.Tensor ~ [sum(seq_lens), n_bits]): Bit scores.

        Returns:
            torch.Tensor ~ sum(seq_lens): Bit predictions.
        """
        s_bit[:, self.bit_conf.special_indices] = s_bit.min()-1
        return s_bit.argmax(-1)
    
    
    def bit_pred(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Bit prediction.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_len), hidden_size]): Contextualized word embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            torch.Tensor ~ sum(seq_lens): Predicted bits per token.
        """
        return self.bit_post(self.bit(embed)[mask])