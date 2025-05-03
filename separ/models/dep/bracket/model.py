from typing import Optional, List, Tuple 
from torch import nn 
import torch 

from separ.modules import FFN 
from separ.model import Model 
from separ.utils import Config 

class BracketDependencyModel(Model):
    def __init__(
        self,
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        bracket_conf: Config,
        rel_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.bracket_conf, self.rel_conf = bracket_conf, rel_conf
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, bracket_conf, rel_conf]
        
        self.bracket = FFN(self.hidden_size, bracket_conf.vocab_size)
        self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(self, words: torch.Tensor, feats: List[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: 
                s_bracket ~ [sum(seq_lens), n_brackets]: Flattened bracket scores per token.
                s_rel ~ [sum(seq_lens), n_rels]: Flattened REL scores per token.
        """
        embed = self.encode(words, *feats)[mask]
        return self.bracket(embed), self.rel(embed)
    
    def loss(self, s_bracket: torch.Tensor, s_rel: torch.Tensor, brackets: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_bracket (torch.Tensor ~ [sum(seq_lens), n_brackets]): Flattened index scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened relation scores per token.
            brackets (torch.Tensor ~ sum(seq_lens)): Flattened brackets per token.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened rel per token.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return self.criteria(s_bracket, brackets) + self.criteria(s_rel, rels)

    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        brackets: torch.Tensor, 
        rels: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control (evaluation + prediction) step.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            brackets (torch.Tensor ~ sum(seq_lens)): Flattened brackets per token.
            rels (torch.Tensor ~ sum(seq_lens)): Flattened rel per token.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        s_bracket, s_rel = self.forward(words, feats, mask)
        loss = self.loss(s_bracket, s_rel, brackets, rels)
        return loss, *self.postprocess(s_bracket, s_rel)
    
    def postprocess(self, s_bracket: torch.Tensor, s_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores post-processing to suppress non-valid indices.

        Args:
            s_bracket (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened bracket scores per token.
            s_rel (torch.Tensor ~ [sum(seq_lens), n_rels]): Flattened REL scores per token.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Post-processed predictions.
        """
        s_bracket[:, self.bracket_conf.special_indices] = s_bracket.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        return s_bracket.argmax(-1), s_rel.argmax(-1)
    
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
            bracket_preds (torch.Tensor ~ sum(seq_lens)]): Index predictions.
            rel_preds (torch.Tensor ~ sum(seq_lens)]): Relation predictions.
        """
        return self.postprocess(*self.forward(words, feats, mask))