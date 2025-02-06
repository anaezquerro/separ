from torch import nn 
import torch 
from trasepar.modules.ffn import FFN 
from trasepar.modules.embed import SinusoidalPositionalEmbedding

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Attention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        dropout: float = 0.0, 
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(input_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Forward pass for the attention module.

        Args:
            x (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Inputs.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Attention mask.

        Returns:

        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = mask[:, None, None, :]
        mask = (mask != 1) * -10000
        attn += mask.to(torch.float32)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop(x)
        return x

class DiTBlock(nn.Module):
    """
    Transformer block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_heads: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(input_size, hidden_size, num_heads=num_heads, qkv_bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FFN(hidden_size, hidden_size, activation=nn.GELU())
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


class DiTFinal(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 1152,
        num_layers: int = 6,
        num_heads: int = 16,
        max_len: int = 512
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads

        self.x_embed = nn.Linear(input_size, hidden_size)
        self.t_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_size),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.p_embed = nn.Embedding(max_len, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.final = DiTFinal(hidden_size, input_size)
        self.reset_parameters()

    def reset_parameters(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embed[1].weight, std=0.02)
        nn.init.normal_(self.t_embed[3].weight, std=0.02)

    def forward(self, noise: torch.Tensor, time: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass.
        
        Args:
            noise (torch.Tensor ~ [batch_size, max(seq_lens), n_bits]): Input noise.
            time (torch.Tensor ~ batch_size): Diffusion timesteps.
            cond (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Conditional embeddings.
            
        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens), n_bits]: Denoised output.
        """
        pos = torch.arange(noise.shape[1], dtype=torch.long, device=noise.device).unsqueeze(dim=0)
        x = self.x_embed(noise) + self.p_embed(pos) # [bsz, len, hidden_size]
        t = self.t_embed(time).unsqueeze(dim=1).expand(-1, x.shape[1], -1) # [bsz, len, hidden_size]
        c = t + cond
        for block in self.blocks:
            x = block(x, c, mask)  # (N, T, D)
        x = self.final(x, c)
        return x
