from typing import List 
from torch import nn 
import torch 

from separ.model import Model 
from separ.utils import Config 
from separ.modules import FFN 


class TagModel(Model):
    def __init__(
        self, 
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Config, 
        char_conf: Config,
        *target_confs: List[Config]
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        for conf in target_confs:
            self.__setattr__(conf.name.lower(), FFN(self.hidden_size, conf.vocab_size))
        self.target_confs = target_confs
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(
        self,
        words: torch.Tensor, 
        feats: List[torch.Tensor],
        *masks: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Forward pass for the tagging model.

        Args:
            words (torch.Tensor ~ [batch_size, max_len]): Batch of words.
            feats (List[torch.Tensor ~ [batch_size, max_len]]): List of batched features.
            mask (torch.Tensor ~ [batch_size, max_len]): Padding mask.

        Returns:
            List[torch.Tensor ~ [sum(seq_lens), num_tags]]: List of flattened scores.
        """
        embed = self.encode(words, *feats)
        return [self.__getattr__(conf.name.lower())(embed[mask]) for conf, mask in zip(self.target_confs, masks)]
    
    def loss(
        self,
        scores: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> torch.Tensor:
        """Computes cross-entropy loss.

        Args:
            scores (List[torch.Tensor ~ [sum(seq_lens), num_tags]]): List of flattened scores.
            targets (List[torch.Tensor ~ sum(seq_lens)]): List of real tags.

        Returns:
            torch.Tensor.
        """
        return sum(map(self.criteria, scores, targets))
    
    
    def predict(self, scores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prediction of the tagging model.

        Args:
            words (torch.Tensor ~ [batch_size, max_len]): Batch of words.
            feats (List[torch.Tensor ~ [batch_size, max_len]]): List of batched features.
            mask (torch.Tensor ~ [batch_size, max_len]): Padding mask.

        Returns:
            List[torch.Tensor ~ sum(seq_lens)]: List of predicted tags.
        """
        preds = []
        for conf, score in zip(self.target_confs, scores):
            score = torch.nan_to_num(score, 0)
            score[:, conf.special_indices] = score.min()-1
            preds.append(score.argmax(-1))
        return preds 
    