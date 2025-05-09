from typing import List, Tuple
from torch import nn 
from transformers import AutoModel
import torch 
from torch.nn.utils.rnn import pad_sequence

from separ.utils import split, max_len

def agg(embed: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    end = lens.cumsum(0).tolist()
    start = [0] + end[:-1]
    return torch.stack([embed[s:e].mean(0) for s, e in zip(start, end)])

class PretrainedEmbedding(nn.Module):
    DEFAULT_MAX_LEN = 512
    
    def __init__(
            self, 
            pretrained: str, 
            pad_index: int, 
            vocab_size: int,
            finetune: bool,
            **_
        ):
        super().__init__()
        self.pretrained = pretrained
        self.finetune = finetune
        self.embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.embed.resize_token_embeddings(vocab_size)
        self.embed_size = self.embed.config.hidden_size
        self.pad_index = pad_index
        self.max_len = 512
        
    def __repr__(self) -> str:
        return f'PretrainedEmbedding(pretrained={self.pretrained}, embed_size={self.embed_size}, finetune={self.finetune})'

    def forward(self, x: Tuple[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        words, lens = x
        mask = (words != self.pad_index).to(torch.int32)
        if words.shape[-1] > self.max_len:
            seq_len = self.max_len//2
            embed = torch.cat([
                self.embed(words[:, i:(i+seq_len)], attention_mask=mask[:, i:(i+seq_len)]).last_hidden_state
                for i in range(0, words.shape[1], seq_len)
            ], 1)
        else:
            embed = self.embed(words, attention_mask=mask).last_hidden_state
        embed = list(map(agg, embed.unbind(0), lens))
        return pad_sequence(embed, batch_first=True)
    


