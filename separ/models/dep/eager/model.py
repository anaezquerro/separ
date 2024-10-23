from typing import Optional, List, Tuple 
from torch import nn 
import torch 

from separ.model import Model
from separ.modules import FFN
from separ.utils import Config

class ArcEagerDependencyModel(Model):
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        act_conf: Config,
        rel_conf: Config,
        n_stack: int,
        n_buffer: int
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf, act_conf=act_conf, rel_conf=rel_conf)
        self.action = FFN(enc_conf.hidden_size*(n_stack + n_buffer), act_conf.vocab_size)
        self.rel = FFN(enc_conf.hidden_size*(n_stack + n_buffer), rel_conf.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            states (List[torch.Tensor ~ [tr_len, n_stack+n_buffer]] ~ batch_size): List of indices 
                of each tokens used to represent a state.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: 
                s_action ~ [sum(tr_lens), n_actions]: Flattened action scores per state.
                s_rel ~ [sum(tr_lens), n_rels]: Flattened REL scores per state.
        """
        embed = self.encode(words, *feats)
        state_embed = torch.cat([embed[i, state].flatten(-2) for i, state in enumerate(states)])
        s_action = self.action(state_embed)
        s_rel = self.rel(state_embed)
        return s_action, s_rel
        
    def loss(self, s_action: torch.Tensor, s_rel: torch.Tensor, actions: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        """Loss computation.

        Args:
            s_label (torch.Tensor ~ [sum(tr_lens), n_actions]): Flattened action scores per state.
            s_rel (torch.Tensor ~ [sum(tr_lens), n_rels]): Flattened REL scores per per state.
            labels (torch.Tensor ~ sum(tr_lens)): Gold actions per state.
            rels (torch.Tensor ~ sum(tr_lens)): Gold RELs per state.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        mask = rels != self.rel_conf.pad_index
        return self.criteria(s_action, actions) + self.criteria(s_rel[mask], rels[mask])
    
    def predict(self, embed: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Label and REL batch prediction for a single state. 

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word 
                embeddings.
            states (List[torch.Tensor ~ [1, n_stack+n_buffer]] ~ batch_size): List of indices 
                of each tokens used to represent a state.
                
        Returns:
            action_preds ~ [batch_size, n_actions]: Ordered action prediction..
            rel_preds ~ batch_size: REL prediction.
        """
        state_embed = torch.cat([embed[i, state].flatten(-2) for i, state in enumerate(states)])
        s_action = self.action(state_embed)
        s_rel = self.rel(state_embed)
        action_preds = s_action.argsort(-1, descending=True)
        rel_preds = s_rel.argmax(-1)
        return action_preds, rel_preds
        
    def control(self, words: torch.Tensor, feats: List[torch.Tensor], states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode(words, *feats)
        state_embed = torch.cat([embed[i, state].flatten(-2) for i, state in enumerate(states)])
        s_action = self.action(state_embed)
        s_rel = self.rel(state_embed)
        return embed, s_action, s_rel

        
    