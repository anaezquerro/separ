from typing import Optional, Tuple, List 
from torch import nn
import torch 

from separ.models.tag.model import TagModel
from separ.utils import Config
from separ.modules import FFN 

class SemanticSLModel(TagModel):
    def __init__(
        self, 
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        label_conf: Config,
        rel_conf: Config
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf, label_conf, rel_conf)
        if not rel_conf.join_rels:
            self.head = FFN(self.hidden_size, self.hidden_size//2)
            self.dep = FFN(self.hidden_size, self.hidden_size//2)
            
    @property
    def label(self) -> FFN:
        return self.__getattr__(self.target_confs[0].name.lower())
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        mask: torch.Tensor,
        matrices: Optional[torch.Tensor], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if matrices is None:
            return super().forward(words, feats, mask)
        else:
            embed = self.encode(words, *feats)
            return self.label(embed[mask]), self.score_rel(embed, matrices, int(mask.sum()))
        
    def score_rel(self, embed: torch.Tensor, matrices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Relation scoring.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            matrices (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Batch of adjacent matrices.
            batch_size (int): Batch-size.

        Returns:
            torch.Tensor ~ [sum(matrices), n_rels]: Relation scores per arc.
        """
        b, deps, heads = matrices.nonzero(as_tuple=True)
        s_rel = []
        for i in range(0, len(b), batch_size):
            head_embed = self.head(embed[b[i:(i+batch_size)], heads[i:(i+batch_size)]])
            dep_embed = self.dep(embed[b[i:(i+batch_size)], deps[i:(i+batch_size)]])
            s_rel.append(self.rel(torch.cat([head_embed, dep_embed], -1)))
        return torch.cat(s_rel, 0) if len(s_rel) > 0 else torch.tensor([], device=embed.device)
    
    def loss(self, scores: Tuple[torch.Tensor, torch.Tensor], targets: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy loss.

        Args:
            s_label (torch.Tensor ~ [sum(seq_lens), n_labels]): Concatenated label scores.
            s_rel (torch.Tensor ~ [sum(seq_lens) | sum(matrices), n_rels]): Concatenated relation scores.
            labels (torch.Tensor ~ sum(seq_lens)): Concatenated labels.
            rels (torch.Tensor ~ sum(seq_lens) | sum(matrices)): Concatenated relations.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        s_label, s_rel, labels, rels = *scores, *targets
        loss = self.criteria(s_label, labels.to(torch.long))
        if len(rels) > 0:
            loss += self.criteria(s_rel, rels.to(torch.long))
        return loss 
    
    def predict_label(self, embed: torch.Tensor) -> torch.Tensor:
        """Label prediction. 

        Args:
            embed (torch.Tensor ~ [sum(seq_lens), hidden_size]): Contextualized word embeddings.

        Returns:
            torch.Tensor: Label prediction.
        """
        s_label = self.label(embed)
        for i in self.target_confs[0].special_indices:
            s_label[:, i] = s_label.min()-1 
        return s_label.argmax(-1)
    
    def predict_rel(self, embed: torch.Tensor, matrices: torch.Tensor, batch_size: int) -> torch.Tensor:
        s_rel = self.score_rel(embed, matrices, batch_size)
        if len(s_rel) == 0:
            return s_rel
        for i in self.target_confs[-1].special_indices:
            s_rel[:, i] = s_rel.min()-1
        return s_rel.argmax(-1)