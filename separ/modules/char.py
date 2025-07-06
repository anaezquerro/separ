from torch import nn 
from torch.nn.utils.rnn import PackedSequence, pad_sequence
import torch 

from separ.modules.ffn import FFN 

class CharLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=False, num_layers=1)
        self.ffn = FFN(hidden_size*2, hidden_size)
        self.reset_parameters()
        
    def forward(self, batch: list[PackedSequence]) -> torch.Tensor:
        embed = [self.embed(x) for x in batch]
        return self.ffn(pad_sequence(embed, batch_first=True))
        
    def embed(self, x: PackedSequence) -> torch.Tensor:
        return torch.cat(self.lstm(x)[1], -1).squeeze(0)
        
    def __repr__(self) -> str:
        return f'CharLSTM(dim={self.hidden_size})'

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)