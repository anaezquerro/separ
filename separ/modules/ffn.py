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
        self.mlp = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(input_size, output_size),
            activation
        )
        self.dropout = dropout 
        self.output_size = output_size
        self.input_size = input_size
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            x = x.contiguous().view(-1, x.shape[-1])
            return self.mlp(x).view(batch_size, seq_len, self.output_size)
        elif len(x.shape) == 2:
            return self.mlp(x)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return f'FFN(n_in={self.input_size}, n_out={self.output_size}, act={self.mlp[-1].__class__.__name__}, dropout={self.dropout})'

    def reset_parameters(self):
        nn.init.orthogonal_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)