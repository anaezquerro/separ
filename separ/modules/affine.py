from __future__ import annotations
import torch
import torch.nn as nn

from separ.modules.ffn import FFN
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
        decompose (bool):
            If ``True``, represents the weight as the product of 2 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_proj: int | None = None,
        dropout: float = 0.0,
        scale: int = 0,
        bias_x: bool = True,
        bias_y: bool = True,
        decompose: bool = False
    ) -> Biaffine:
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        
        if n_proj is not None:
            self.mlp_x, self.mlp_y = FFN(n_in, n_proj, dropout), FFN(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(torch.zeros(n_out, self.n_model + bias_x, self.n_model + bias_y))
        else:
            self.weight = [(nn.Parameter(torch.zeros(n_out, self.n_model + bias_x)),
                            nn.Parameter(torch.zeros(n_out, self.n_model + bias_y)))]


    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y = self.mlp_x(x), self.mlp_y(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wy = torch.einsum('byj,oj->boy', y, self.weight[1])
            s = torch.einsum('box,boy->boxy', wx, wy)
        else:
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
            # s = torch.stack(
                # [biaf(x[b], self.weight[i], y[b]) for b in range(x.shape[0]) for i in range(self.n_out)]
            # ).reshape(x.shape[0], self.n_out, x.shape[1], x.shape[1])
        return s.squeeze(1) / self.n_in ** self.scale


def biaf(X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Biaffine product.

    Args:
        X (torch.Tensor ~ [seq_len, x_dim]): First input.
        W (torch.Tensor ~ [x_dim, y_dim]): Weight matrix.
        Y (torch.Tensor ~ [seq_len, y_dim]): Second input.

    Returns:
        torch.Tensor ~ [seq_len, seq_len]: Score matrix.
    """
    return X @ W @ Y.T
    