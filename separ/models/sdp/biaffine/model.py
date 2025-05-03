from typing import Optional, List, Tuple 
from torch import nn 
import torch 

from separ.model import Model 
from separ.utils import Config
from separ.modules import FFN, Biaffine

class BiaffineSemanticModel(Model):

    def __init__(
        self,
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        rel_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.arc = Biaffine(self.hidden_size, 2, bias_x=True, bias_y=True)
        self.rel = Biaffine(self.hidden_size, rel_conf.vocab_size, bias_x=True, bias_y=True)
        self.head_arc = FFN(self.hidden_size, self.hidden_size)
        self.dep_arc = FFN(self.hidden_size, self.hidden_size)
        self.head_rel = FFN(self.hidden_size, self.hidden_size)
        self.dep_rel = FFN(self.hidden_size, self.hidden_size)
        self.rel_conf = rel_conf 
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, rel_conf]
        
    def forward(self, words: torch.Tensor, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_lens)] if not pretrained.
                ~ [batch_size, max(seq_lens), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_lens)]] ~ n_feats): Input features.

        Returns:
            s_arc ~ [batch_size, max(seq_lens), max(seq_lens), 2]: Arc scores.
            s_rel ~ [batch_size, max(seq_lens), max(seq_lens), n_rels]: Label scores.
        """
        embed = self.encode(words, *feats)
        head_arc, dep_arc = self.head_arc(embed), self.dep_arc(embed)
        head_rel, dep_rel = self.head_rel(embed), self.dep_rel(embed)
        s_arc = self.arc(dep_arc, head_arc).permute(0, 2, 3, 1)
        s_rel = self.rel(dep_rel, head_rel).permute(0, 2, 3, 1)
        return s_arc, s_rel
    
    def loss(self, s_arc: torch.Tensor, s_rel: torch.Tensor, arcs: torch.Tensor, rels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_arc (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens), 2]): Arc scores.
            s_rel (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens), n_rels]): Label scores.
            arcs (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Arcs.
            rels (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Labels.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        arc_mask = arcs.to(torch.bool) & mask 
        arc_mask[:, 0] = False
        # arc_mask.diagonal(dim1=-2, dim2=-1).zero_()
        arc_loss = self.criteria(s_arc[mask], arcs[mask])
        rel_loss = self.criteria(s_rel[arc_mask], rels[arc_mask])
        return 3*arc_loss + rel_loss 
    
    def predict(self, words: torch.Tensor, feats: List[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Arc prediction.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_lens)] if not pretrained.
                ~ [batch_size, max(seq_lens), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_lens)]] ~ n_feats): Input features.
            mask (torch.Tensor ~ [batch_szize, max(seq_lens)]): Padding mask.

        Returns:
            arc_preds (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Arc scores.
            rel_preds (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Label scores.
        """
        s_arc, s_rel = self.forward(words, feats)
        arc_preds = s_arc.argmax(-1)
        arc_preds[~mask] = 0 
        arc_preds[:, 0] = False
        # arc_preds.diagonal(dim1=-2, dim2=-1).zero_()
        s_rel[:, :, :, self.rel_conf.special_indices] = s_rel.min()-1
        rel_preds = s_rel.argmax(-1)
        return arc_preds, rel_preds
    
    def evaluate(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        arcs: torch.Tensor, 
        rels: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluation.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_lens)] if not pretrained.
                ~ [batch_size, max(seq_lens), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_lens)]] ~ n_feats): Input features.
            arcs (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Arcs.
            rels (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Labels.
            mask (torch.Tensor ~ [batch_szize, max(seq_lens)]): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss and predictions.
        """
        s_arc, s_rel = self(words, feats)
        loss = self.loss(s_arc, s_rel, arcs, rels, mask)
        arc_preds = s_arc.argmax(-1)
        arc_preds[~mask] = 0 
        arc_preds[:, 0] = 0
        # arc_preds.diagonal(dim1=-2, dim2=-1).zero_()
        s_rel[:, :, :, self.rel_conf.special_indices] = s_rel.min()-1
        rel_preds = s_rel.argmax(-1)
        return loss, arc_preds, rel_preds



    

