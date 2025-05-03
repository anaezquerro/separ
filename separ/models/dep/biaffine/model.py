from typing import Tuple, List, Optional
import torch 
from torch import nn 

from separ.data.struct import MST 
from separ.model import Model 
from separ.utils import Config, pad2D 
from separ.modules import FFN, Biaffine


class BiaffineDependencyModel(Model):
    
    def __init__(
        self,
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config],
        char_conf: Optional[Config],
        rel_conf: Config
    ):
        enc_conf.hidden_size = 400
        super().__init__(enc_conf, word_conf, tag_conf, char_conf)
        self.arc = Biaffine(self.hidden_size, 1, bias_x=True, bias_y=False)
        self.rel = Biaffine(self.hidden_size, rel_conf.vocab_size, bias_x=True, bias_y=True)
        self.head_arc = FFN(self.hidden_size, self.hidden_size)
        self.dep_arc = FFN(self.hidden_size, self.hidden_size)
        self.head_rel = FFN(self.hidden_size, self.hidden_size)
        self.dep_rel = FFN(self.hidden_size, self.hidden_size)
        self.rel_conf = rel_conf 
        self.criteria = nn.CrossEntropyLoss()
        self.confs = [enc_conf, word_conf, tag_conf, char_conf, rel_conf]
        
    def forward(self, words: torch.Tensor, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.encode(words, *feats)
        head_arc, dep_arc = self.head_arc(embed), self.dep_arc(embed)
        head_rel, dep_rel = self.head_rel(embed), self.dep_rel(embed)
        s_arc = self.arc(dep_arc, head_arc)
        s_rel = self.rel(dep_rel, head_rel).permute(0, 2, 3, 1)
        return s_arc, s_rel
    
    def loss(self, s_arc: torch.Tensor, s_rel: torch.Tensor, arcs: torch.Tensor, rels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        heads = arcs[mask].argmax(-1)
        arc_loss = self.criteria(s_arc[mask], heads.to(torch.long))
        rel_loss = self.criteria(s_rel[mask, heads], rels[mask, heads])
        return arc_loss + rel_loss 
    
    def predict(self, words: torch.Tensor, feats: List[torch.Tensor], mask0: torch.Tensor, mask1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_arc, s_rel = self.forward(words, feats)
        s_arc = self.postprocess(s_arc, mask1)
        lens = mask0.sum(-1).tolist()
        arc_preds = pad2D([MST(score[:(l+1), :(l+1)]) for score, l in zip(s_arc, lens)]).to(torch.bool)
        s_rel = s_rel[arc_preds]
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        rel_preds = s_rel.argmax(-1)
        return arc_preds.to(int)[mask0].argmax(-1), rel_preds
    
    def postprocess(self, s_arc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s_arc[~mask] = s_arc.min()-1  # remove padding 
        s_arc[:, 0] = s_arc.min()-1 # remove incoming arcs to root 
        s_arc.diagonal(dim1=-2, dim2=-1).fill_(s_arc.min()-1) # remove cycles of length 1
        return s_arc
    
    def control(
        self, 
        words: torch.Tensor, 
        feats: List[torch.Tensor], 
        arcs: torch.Tensor, 
        rels: torch.Tensor, 
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lens = mask0.sum(-1).tolist()
        s_arc, s_rel = self.forward(words, feats)
        
        # compute loss
        heads = arcs[mask0].argmax(-1)
        loss = self.criteria(s_arc[mask0], heads.to(torch.long)) + self.criteria(s_rel[mask0, heads], rels[mask0, heads])
        
        # compute rels
        s_arc = self.postprocess(s_arc, mask1)
        head_preds = torch.cat([MST(score[:(l+1), :(l+1)]).to(int).argmax(-1)[1:] for score, l in zip(s_arc, lens)])
        try:
            s_rel = s_rel[mask0, head_preds]
        except:
            print('error')
        s_rel[:, self.rel_conf.special_indices] = s_rel.min()-1
        rel_preds = s_rel.argmax(-1)
        if len(head_preds)  != len(rel_preds):
            print('error')
        return loss, head_preds, rel_preds