from __future__ import annotations 
from torch import nn 
import torch 


class SharedDropout(nn.Module):
    r"""
    :class:`SharedDropout` differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.

    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.

    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x = torch.ones(batch_size, seq_len, hidden_size)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p: float = 0.5, batch_first: bool = True) -> SharedDropout:
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            A tensor with the same shape as `x`.
        """

        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.FloatTensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)
