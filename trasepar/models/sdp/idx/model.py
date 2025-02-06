from typing import Optional, List, Tuple
from torch import nn 
import torch 

from trasepar.model import Model 
from trasepar.utils import Config
from trasepar.modules import FFN  

class IndexSemanticModel(Model):
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
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, index_conf, rel_conf]
        
        self.index = FFN(self.hidden_size, index_conf.vocab_size)
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
                s_index ~ [sum(seq_lens), n_indexes]: Flattened label scores per token.
                s_rel ~ [sum(matrices), n_rels]: Flattened REL scores per gold arc.
        """
        embed = self.encode(words, *feats)
        return self.index(embed[mask]), self.rel_score(embed, matrices, int(mask.sum()))
    
    def loss(self, s_index: torch.Tensor, s_rel: torch.Tensor, indexes: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Flattened index scores per token.
            s_rel (torch.Tensor, [sum(matrices), n_rels]): Flattened relation scores per gold arc.
            indexes (torch.Tensor ~ sum(seq_lens)): Gold indexes per token.
            rels (torch.Tensor ~ sum(matrices)): Gold RELs per gold arc.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        loss = self.criteria(s_index, indexes.to(torch.long))
        if len(rels) > 0:
            loss += self.criteria(s_rel, rels.to(torch.long))
        return loss 
    
    def control(
        self, 
        words: torch.Tensor,
        feats: List[torch.Tensor],
        indexes: torch.Tensor, 
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
            indexes (torch.Tensor ~ sum(seq_lens)): Gold indexes per token.
            rels (torch.Tensor ~ sum(matrices)): Gold relations per token.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched and padded adjacent matrices. 
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            loss (torch.Tensor): Cross-entropy loss.
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            index_preds (torch.Tensor ~ max(seq_lens)): Flattened label predictions.
            s_rel (torch.Tensor ~ [max(seq_lens), n_rels]): Flattened relation scores.
        """
        embed = self.encode(words, *feats)
        s_index = self.index(embed)[mask]
        s_rel = self.rel_score(embed, matrices, int(mask.sum()))
        loss = self.loss(s_index, s_rel, indexes, rels)
        return loss, embed, self.index_post(s_index), s_rel
    
    def index_post(self, s_index: torch.Tensor) -> torch.Tensor:
        """Index scores post-processing and prediction.

        Args:
            s_index (torch.Tensor ~ [sum(seq_lens), n_indexes]): Index scores.

        Returns:
            torch.Tensor: Index predictions.
        """
        s_index[:, self.index_conf.special_indices] = s_index.min()-1
        index_preds = s_index.argmax(-1)
        for i in self.index_conf.special_indices:
            index_preds[index_preds == i] = self.index_conf.most_frequent 
        return index_preds
    
    
    def index_pred(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Index prediction.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_len), hidden_size]): Contextualized word embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_len)]): Padding mask.

        Returns:
            torch.Tensor ~ sum(seq_lens): Predicted indexes per token.
        """
        return self.index_post(self.index(embed)[mask])
    
    def rel_score(self, embed: torch.Tensor, matrices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Computes relation scores.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Padded and batched adjacent matrices. 
            batch_size (int): Batch size to iterate over each entry.

        Returns:
            torch.Tensor ~ [sum(matrices), n_rels]: Flattened relation scores.
        """
        b, deps, heads = matrices.nonzero(as_tuple=True)
        s_rel = []
        for i in range(0, len(b), batch_size):
            head_embed = self.head(embed[b[i:(i+batch_size)], heads[i:(i+batch_size)]])
            dep_embed = self.dep(embed[b[i:(i+batch_size)], deps[i:(i+batch_size)]])
            s_rel.append(self.rel(torch.cat([head_embed, dep_embed], -1)))
        return torch.cat(s_rel, 0) if len(s_rel) > 0 else torch.tensor([], device=embed.device)
    
    def rel_post(self, s_rel: torch.Tensor) -> torch.Tensor:
        """Dependency relation scores post-processing.

        Args:
            s_rel (torch.Tensor ~ [sum(matrices), n_rels]): Dependency relation scores.

        Returns:
            torch.Tensor ~ sum(matrices): Dependency relation predictions.
        """
        if len(s_rel) == 0:
            return torch.tensor([])
        else:
            s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
            return s_rel.argmax(-1)
        
    def rel_pred(self, embed: torch.Tensor, matrices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Dependency relation prediction.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Padded and batched adjacent matrices. 
            batch_size (int): Batch size to iterate over each entry.

        Returns:
            torch.Tensor ~ sum(matrices): Dependency relation predictions.
        """
        return self.rel_post(self.rel_score(embed, matrices, batch_size))
        
