"""Module adapted from https://github.com/tkipf/pygcn"""


import torch, math 
from typing import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn


class GCNLayer(Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super(GCNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.FloatTensor(input_size, output_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Forward pass of the graph convolutional layer.

        Args:
            x (torch.Tensor ~ [seq_len, input_size]): Input batch.
            m (torch.Tensor ~ [seq_len, seq_len]): Adjacent matrix.

        Returns:
            torch.Tensor ~ [batch_size, seq_len, output_size]: Output embeddings.
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_size) + ' -> ' \
               + str(self.output_size) + ')'

class GCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1, 
        act: nn.Module = nn.LeakyReLU(0.1)
    ):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout = dropout
        self.act = act 

        self.layers = nn.ModuleList()
        in_dim, out_dim = input_size, hidden_size 
        for i in range(num_layers):
            self.layers.append(GCNLayer(in_dim, out_dim))
            in_dim = out_dim 
            if i + 1 == num_layers: # last layer 
                out_dim = self.output_size
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN.

        Args:
            x (torch.Tensor ~ [batch_size, seq_len, input_size]): Input batch.
            m (torch.Tensor ~ [batch_size, seq_len, seq_len]): Adjacent matrix.

        Returns:
            torch.Tensor ~ [batch_size, seq_len, output_size]: Output embeddings.
        """
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, m)
        return self.act(x)