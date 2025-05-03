
from typing import Optional, List, Tuple
from torch import nn 
import torch 

from separ.modules import FFN 
from separ.model import Model 
from separ.utils import Config 

class Bit4DependencyModel(Model):
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        bit_conf: Config,
        rel_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.bit_conf = bit_conf 
        self.rel_conf = rel_conf 
        self.bit = FFN(self.hidden_size, bit_conf.vocab_size)
        self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, bit_conf, rel_conf]
        
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
                s_bit ~ [sum(seq_lens), n_bits]: Flattened bit scores per token.
                s_rel ~ [sum(seq_lens), n_rels]: Flattened relation scores per token.
        """
        embed = self.encode(words, *feats)[mask]
        return self.bit(embed), self.rel(embed) 
    
    def loss(self, s_bit: torch.Tensor, s_rel: torch.Tensor, bits: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_bit (torch.Tensor ~ [sum(seq_lens), n_bits]): Flattened bit scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.
            bits (torch.Tensor ~ sum(seq_lens)): Flattened batch of bits.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened batch of rels.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return self.criteria(s_bit, bits) + self.criteria(s_rel, rels)
    
    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor],
        bits: torch.Tensor, 
        rels: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (evaluation + prediction) step.

        Args:
            words (torch.Tensor): Padded batch of input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            bits (torch.Tensor ~ sum(seq_lens)): Flattened batch of bits.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened batch of rels.
            mask (torch.Tensor): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        scores = self.forward(words, feats, mask)
        loss = self.loss(*scores, bits, rels)
        return loss, *self.postprocess(*scores)
    
    def postprocess(self, s_bit: torch.Tensor, s_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores post-processing to suppress non-valid indices.

        Args:
            s_bit (torch.Tensor ~ [sum(seq_lens), n_bits]): Flattened bit scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.
            
        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: Post-processed predictions.
        """
        s_bit[:, self.bit_conf.special_indices] = s_bit.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        return s_bit.argmax(-1), s_rel.argmax(-1)
    
    def predict(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bit and REL prediction.

        Args:
            words (torch.Tensor): Padded batch of input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns: 
            bit_preds (torch.Tensor ~ sum(seq_lens)]): Bit predictions.
            rel_preds (torch.Tensor ~ sum(seq_lens)]): Relation predictions.
        """
        return self.postprocess(*self.forward(words, feats, mask))
    