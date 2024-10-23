import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.LeakyReLU(0.1)
    ):
        super().__init__()
        self.mlp = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = activation
        self.dropout = dropout 
        self.output_size = output_size
        self.input_size = input_size
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            x = x.contiguous().view(-1, x.shape[-1])
            x = self.act(self.mlp(self.drop(x)))
            return x.view(batch_size, seq_len, self.output_size)
        elif len(x.shape) == 2:
            return self.act(self.mlp(self.drop(x)))
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return f'FFN(n_in={self.input_size}, n_out={self.output_size}, act={self.act.__class__.__name__}, dropout={self.dropout})'

    def reset_parameters(self):
        nn.init.orthogonal_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)