from torch import nn 
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
from typing import Union 
import torch, math
from einops import rearrange

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, pad_index: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, pad_index)
        self.vocab_size, self.embed_size, self.pad_index = vocab_size, embed_size, pad_index
        self.reset_parameters()

    def __repr__(self) -> str:
        return f'Embedding(vocab_size={self.vocab_size}, embed_size={self.embed_size}, pad_index={self.pad_index})'
    
    def reset_parameters(self):
        nn.init.normal_(self.embed.weight, 0, self.embed.embedding_dim ** -0.5)
        nn.init.zeros_(self.embed.weight[self.pad_index])
    
    def forward(self, inputs: Union[torch.Tensor, PackedSequence]) -> Union[torch.Tensor, PackedSequence]:
        if isinstance(inputs, torch.Tensor):
            return self.embed(inputs)
        else:
            _, lens = pad_packed_sequence(inputs, batch_first=True)
            embed = self.embed(inputs.data).split(lens.tolist())
            return pack_sequence(embed, enforce_sorted=False)
            