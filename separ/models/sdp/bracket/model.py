
from typing import Optional, List, Tuple
from torch import nn 
import torch 

from separ.modules import FFN 
from separ.utils import Config
from separ.models.sdp.idx.model import IndexSemanticModel
from separ.model import Model

class BracketSemanticModel(IndexSemanticModel):
    def __init__(
        self, 
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        bracket_conf: Config,
        rel_conf: Config
    ):
        Model.__init__(self, enc_conf, word_conf, tag_conf, char_conf)
        self.bracket_conf = bracket_conf
        self.rel_conf = rel_conf
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, bracket_conf, rel_conf]
        
        self.bracket = FFN(self.hidden_size, bracket_conf.vocab_size)
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
                s_bracket ~ [sum(seq_lens), n_brackets]: Flattened label scores per token.
                s_rel ~ [sum(matrices), n_rels]: Flattened REL scores per gold arc.
        """
        embed = self.encode(words, *feats)
        return self.bracket(embed)[mask], self.rel_score(embed, matrices, int(mask.sum()))
    
    def loss(self, s_bracket: torch.Tensor, s_rel: torch.Tensor, brackets: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_bracket (torch.Tensor ~ [sum(seq_lens), n_brackets]): Flattened bracket scores per token.
            s_rel (torch.Tensor, [sum(matrices), n_rels]): Flattened relation scores per gold arc.
            brackets (torch.Tensor ~ sum(seq_lens)): Gold brackets per token.
            rels (torch.Tensor ~ sum(matrices)): Gold RELs per gold arc.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        loss = self.criteria(s_bracket, brackets.to(torch.long)) 
        if len(rels) > 0:
            loss += self.criteria(s_rel, rels.to(torch.long))
        return loss 
        
    def control(
        self, 
        words: torch.Tensor,
        feats: List[torch.Tensor],
        brackets: torch.Tensor, 
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
            brackets (torch.Tensor ~ sum(seq_lens)): Gold brackets per token.
            rels (torch.Tensor ~ sum(matrices)): Gold relations per token.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched and padded adjacent matrices. 
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            loss (torch.Tensor): Cross-entropy loss.
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            bracket_preds (torch.Tensor ~ max(seq_lens)): Flattened label predictions.
            s_rel (torch.Tensor ~ [max(seq_lens), n_rels]): Flattened relation scores.
        """
        embed = self.encode(words, *feats)
        s_bracket = self.bracket(embed)[mask]
        s_rel = self.rel_score(embed, matrices, int(mask.sum()))
        loss = self.loss(s_bracket, s_rel, brackets, rels)
        return loss, embed, self.bracket_post(s_bracket), s_rel
    
    def bracket_post(self, s_bracket: torch.Tensor) -> torch.Tensor:
        """Bracket scores post-processing and prediction.

        Args:
            s_bracket (torch.Tensor ~ [sum(seq_lens), n_brackets]): Bracket scores.

        Returns:
            torch.Tensor ~ sum9seq_lens): Bracket predictions.
        """
        s_bracket[:, self.bracket_conf.special_indices] = s_bracket.min()-1
        return s_bracket.argmax(-1)
        
    
    def bracket_pred(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Bracket prediction.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_len), hidden_size]): Contextualized word embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            torch.Tensor ~ sum(seq_lens): Predicted brackets per token.
        """
        return self.bracket_post(self.bracket(embed)[mask])