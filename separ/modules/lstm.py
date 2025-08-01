from __future__ import annotations 
import torch.nn as nn
import torch


from separ.modules.ffn import FFN 

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
                
    
class xLSTM(nn.Module):
    def __init__(
        self,
        input_size: int, 
        hidden_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        kernel_size: int = 4, 
        block_size: int = 4, 
        num_heads: int = 4,
        context_len: int = 512,
        dropout: float = 0,
        **_
    ):
        super().__init__()
        from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.context_len = context_len
        self.ffn_in = FFN(input_size, hidden_size)
        
        cfg = lambda: xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=kernel_size, qkv_proj_blocksize=block_size, num_heads=num_heads)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=num_heads,
                    conv1d_kernel_size=kernel_size,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_len,
            num_blocks=num_layers*2,
            embedding_dim=hidden_size,
            slstm_at=[i+1 for i in range(0, num_layers*2, 2)],
            dropout=dropout
        )
        self.stack = xLSTMBlockStack(cfg())
        self.ffn_out = FFN(hidden_size, self.output_size) if output_size is not None else nn.Identity()
            
        
    def __repr__(self) -> str:
        return f'xLSTM({self.input_size}, {self.hidden_size}, {self.output_size}, num_layers={self.num_layers})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_out(self.stack(self.ffn_in(x)))
    
    
    
class BixLSTM(nn.Module):
    def __init__(
        self,
        input_size: int, 
        hidden_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        **kwargs
    ):
        super().__init__()
        self.left2right = nn.ModuleList()
        self.left2right.append(xLSTM(input_size, hidden_size//2, num_layers=1, **kwargs))
        self.right2left = nn.ModuleList()
        self.right2left.append(xLSTM(input_size, hidden_size//2, num_layers=1, **kwargs))
        for _ in range(num_layers-1):
            self.left2right.append(xLSTM(hidden_size, hidden_size//2, num_layers=1, **kwargs))
            self.right2left.append(xLSTM(hidden_size, hidden_size//2, num_layers=1, **kwargs))
        self.ffn_out = FFN(hidden_size, self.output_size) if output_size is not None else nn.Identity()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.num_layers = num_layers

    def __repr__(self) -> str:
        return f'BixLSTM({self.input_size}, {self.hidden_size}, {self.output_size}, num_layers={self.num_layers})'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for flayer, rlayer in zip(self.left2right, self.right2left):
            x = torch.cat([flayer(x), rlayer(x.flip(dims=[1])).flip(dims=[1])],dim=-1)
        return self.ffn_out(x)
            
