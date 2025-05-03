from typing import Optional, Tuple, List
from torch import nn 
from torch.nn.functional import cross_entropy
import torch 

from separ.model import Model 
from separ.utils import Config 
from separ.modules import FFN, GCN 

def compute_state(pointers: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Computes the adjacency matrix of each state given Covington's pointers.
    
    Args:
        pointers (torch.Tensor ~ [num_trans, 3]): Pointers.
        matrix (torch.Tensor ~ [batch_size, seq_len, seq_len]): Complete adjacency matrix.
        
    Returns:
        torch.Tensor ~ [num_trans, seq_len, seq_len]: Adjacency matrix at each state.
    """
    states = matrices[pointers[:, 0]].clone()
    for i, (p1, p2) in enumerate(pointers[:, 1:].tolist()):
        states[i, (p2+1):, :] = False
        states[i, :, (p2+1):] = False
        states[i, p2, :(p1+1)] = False 
        states[i, :(p1+1), p2] = False 
    return states.float()

class CovingtonSemanticModel(Model):
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        action_conf: Config,
        rel_conf: Config,
    ):
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.action_conf = action_conf
        self.rel_conf = rel_conf
        self.state = GCN(self.hidden_size, 200)
        self.action = FFN(self.hidden_size*2+200, action_conf.vocab_size)
        self.rel = FFN(self.hidden_size*2+200, rel_conf.vocab_size)
        
    def forward(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        pointers: torch.Tensor,
        matrices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            words (torch.Tensor): Input words.
                ~ [batch_size, max(seq_len)] if not pretrained.
                ~ [batch_size, max(seq_len), fix_len] if pretrained.
            feats (List[torch.Tensor ~ [batch_size, max(seq_len)]] ~ n_feats): Input features.
            pointers (torch.Tensor ~ [num_trans, 3]): Pointers.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched adjacent matrices.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: 
                s_action ~ [sum(tr_lens), n_actions]: Flattened action scores per state.
                s_rel ~ [sum(tr_lens), n_rels]: Flattened REL scores per state.
        """
        embed = self.encode(words, *feats)
        return self.decode(embed, pointers, matrices)
        
    def decode(
        self,
        embed: torch.Tensor,
        pointers: torch.Tensor,
        matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass (decode).

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_len), hidden_size]): Contextualized embeddings.
            pointers (torch.Tensor ~ [num_trans, 3]): Pointers.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched adjacent matrices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Action and relation scores.
        """
        # contextualized embeddings with the GCN 
        # batch_size = embed.shape[0]
        # states = []
        # for i in range(0, pointers.shape[0], batch_size):
        #     points = pointers[i:(i+batch_size)]
        #     states.append(torch.cat([
        #         # arc states
        #         self.state(embed[points[:, 0]], compute_state(points, matrices))[:, :-1].mean(dim=1),
        #         embed[points[:, 0], points[:, 1]], # first pointer
        #         embed[points[:, 0], points[:, 2]] # second pointer
        #     ], dim=-1))
        # state = torch.cat(states)
        # return self.action(state), self.rel(state)
        
        state = torch.cat([
                # arc states
                self.state(embed[pointers[:, 0]], compute_state(pointers, matrices))[:, :-1].mean(dim=1),
                embed[pointers[:, 0], pointers[:, 1]], # first pointer
                embed[pointers[:, 0], pointers[:, 2]] # second pointer
            ], dim=-1)
        s_action = self.action(state)
        s_rel = self.rel(state)
        return s_action, s_rel
    
    def loss(
        self,
        s_action: torch.Tensor, 
        s_rel: torch.Tensor, 
        actions: torch.Tensor,
        rels: torch.Tensor
    ) -> torch.Tensor:
        """Loss computation.

        Args:
            s_label (torch.Tensor ~ [sum(tr_lens), n_actions]): Flattened action scores per state.
            s_rel (torch.Tensor ~ [sum(tr_lens), n_rels]): Flattened REL scores per per state.
            labels (torch.Tensor ~ sum(tr_lens)): Gold actions per state.
            rels (torch.Tensor ~ sum(tr_lens)): Gold RELs per state.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """  
        return cross_entropy(s_action, actions, weight=self.action_conf.weights.to(actions.device)) \
            + cross_entropy(s_rel, rels)
    
    def predict(self, embed: torch.Tensor, pointers: torch.Tensor, matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Label and REL batch prediction for a single state. 

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_len), hidden_size]): Contextualized embeddings.
            pointers (torch.Tensor ~ [num_trans, 3]): Pointers.
            matrices (torch.Tensor ~ [batch_size, max(seq_len), max(seq_len)]): Batched adjacent matrices.
                
        Returns:
            action_preds ~ [batch_size, n_actions]: Ordered action prediction..
            rel_preds ~ batch_size: REL prediction.
        """
        return self.postprocess(*self.decode(embed, pointers, matrices))
    
    def postprocess(self, s_action: torch.Tensor, s_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_action[:, self.action_conf.special_indices] = s_action.min()-1
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        return s_action.argsort(-1, descending=True), s_rel.argmax(-1)
    
    def control(
        self, 
        words: torch.Tensor,
        feats: List[torch.Tensor], 
        mask: torch.Tensor,
        pointers: torch.Tensor,
        matrices: torch.Tensor,
        actions: torch.Tensor,
        rels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode(words, *feats)
        s_action, s_rel = self.decode(embed, pointers, matrices)
        return self.loss(s_action, s_rel[mask], actions, rels[mask]), embed, s_action, s_rel