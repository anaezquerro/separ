from __future__ import annotations 
import torch.nn as nn
import torch
from typing import Optional, Tuple


from separ.modules.ffn import FFN

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        dropout: float = 0.0,
        **_
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.num_layers = num_layers 
        self.dropout = dropout 
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size//(2 if bidirectional else 1), num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True, bias=True)
        if output_size is None:
            self.ffn = activation
        else:
            self.ffn = FFN(hidden_size, output_size, activation=activation)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        y = self.ffn(h)
        return y

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)
                
    def __repr__(self) -> str:
        s = f'LSTM({self.input_size}, {self.hidden_size}'
        if self.output_size:
            s += f', {self.output_size}'
        s += f', n_layers={self.num_layers}, dropout={self.dropout}'
        s += ')'
        return s
                
    
