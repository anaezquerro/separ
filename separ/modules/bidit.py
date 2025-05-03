from torch import nn 
from typing import Iterable, Optional, Tuple 
import torch 

from separ.modules.scheduler import NoiseScheduler
from separ.modules.dit import DiT 
from separ.utils import acc

class BiDiT(nn.Module):
    """Adaptation of the Bit Diffusion Tagger from [Huang et al. (2023)](https://aclanthology.org/2023.findings-emnlp.860/)."""
    
    def __init__(
        self, 
        hidden_size: int,
        sched: NoiseScheduler,
        num_bits: int, 
        max_len: int,
        num_layers: int,
        num_heads: int,
        pred: int,
        steps: int,
        **_
    ): 
        """Initialization of the BiDiT model.

        Args:
            hidden_size (int): Dimension of the decoder embeddings.
            sched (NoiseScheduler): Noise scheduler.
            num_bits (int): Number of bits to transform tags.
            max_len (int): Maximum input length.
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of Transformer heads.
            pred (int): Type of prediction. `0` to directly predict the denoised signal, `-1` to 
                predict the previous latent and `1` to predict the noise.
            loss (int): Loss function (`1` for L1 and `1` for L2).
            steps (int): Number of inference steps.
        """
        assert pred in [0, 1], 'Prediction selection must be in {0, 1}'
        super().__init__()
        self.sched = sched 
        self.T = self.sched.T 
        self.pred = pred 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_bits = num_bits
        self.dim = hidden_size
        self.steps = steps
        self.decoder = DiT(self.num_bits, hidden_size, num_layers=num_layers, max_len=max_len, num_heads=num_heads)
        
    def __repr__(self) -> str:
        return f'BiDiT(num_bits={self.num_bits}, dim={self.dim}, T={self.T}, pred={self.pred},\n\tsteps={self.steps}, num_layers={self.num_layers}, num_heads={self.num_heads})'
            
    def tag2bit(self, tags: torch.Tensor) -> torch.Tensor:
        """Transforms a batch of tags into bits.

        Args:
            tags (torch.Tensor ~ [batch_size, max(seq_lens)]): Batch of tags.
            num_bits (Optional[int]): Number of bits. 
            
        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens), n_bits]: Batch of bits.
        """
        mask = 2**torch.arange(self.num_bits - 1, -1, -1, dtype=torch.int32, device=tags.device)
        bits = (tags.int().unsqueeze(-1) & mask) != 0
        return (bits * 2 - 1).to(torch.float32)
    
    def bit2tag(self, bits: torch.Tensor) -> torch.Tensor:
        """Transforms a batch of bits into tags.
        
        Args: 
            bits (torch.Tensor ~ [batch_size, max(seq_lens), n_bits]): Batch of bits.
            
        Returns: 
            torch.Tensor ~ [batch_size, max(seq_lens)]: Batch of tags.
        """
        mask = 2 ** torch.arange(self.num_bits - 1, -1, -1, dtype=torch.int32, device=bits.device)
        return ((bits > 0)*mask).sum(-1)
    
    def sample_time(self, bits: torch.Tensor) -> torch.Tensor:
        """Sample a random batch of continuous time steps. The first dimension is assumed
        to be the batch size.

        Args:
            bits (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Input bits. 

        Returns:
            torch.Tensor: Random batch of continuous time steps in [1, T].
        """
        batch_size, seq_len, num_bits = bits.shape
        return (torch.rand(batch_size)*(self.T-1)+1).round().view(batch_size, 1, 1).repeat(1, seq_len, num_bits).to(bits.device)
    
    def corrupt(self, bits: torch.Tensor, time: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Corrupts a batch of bits by adding Gaussian noise.

        Args:
            bits (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Batch of bits.
            time (torch.Tensor ~ [batch_size, max(seq_lens), num_bits): Batch of timesteps to corrupt each sequence.
            e (Optional[torch.Tensor], optional): Gaussian noise added to the forward process. If 
                None, no extra-noise is added, so the forward process is deterministic.

        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens), n_bits]: Batch of corrupted bits.
        """
        alpha_bar = self.sched.alpha_bar(time)
        crpt = torch.sqrt(alpha_bar)*bits 
        if e is not None:
            crpt += torch.sqrt(1-alpha_bar)*e 
        return crpt
    
    def forward(self, embed: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Conditional contextualized embeddings.
            tags (torch.Tensor ~ [batch_size, max(seq_lens)]): Tags to predict.
            mask (torch.Tensor): Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and targets.
        """
        bits = self.tag2bit(tags) 
        
        # corrupt bits 
        b = embed.shape[0]
        time = self.sample_time(bits)
        e = torch.randn_like(bits)
        crpt = self.corrupt(bits, time, e)
        preds = self.decoder(crpt, time[:,0,0], embed, mask)
        
        if self.pred == 1: # noise as target
            return preds[mask], e[mask]
        else: # denoised signal as target
            return preds[mask], bits[mask]
        
    def denoise(self, crpt: torch.Tensor, time: torch.Tensor, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Performs one denoising step.
        
        Args:
            crpt (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Corrupted bits.
            time (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Time step of the forward process.
            embed (torch.Tensor ~ [batch_size, max(seq_lens), embed_dim]): Conditional embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.

        Returns:
            torch.Tensor: Previous latent.
        """
        preds = self.decoder(crpt, time[:, 0, 0], embed, mask)

        if self.pred == 1: # predict noise 
            alpha_bar = self.sched.alpha_bar(time)
            alpha_bar_1 = self.sched.alpha_bar(time-self.steps)
            prev = (crpt-torch.sqrt(1-alpha_bar)*preds)/torch.sqrt(alpha_bar)
            prev = torch.sqrt(alpha_bar_1)*prev + torch.sqrt(1-alpha_bar_1)*torch.randn_like(crpt)
        else:
            prev = preds 
        return prev
    
    def predict(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Prediction step.
        
        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), fix_len]): Conditional contextualized embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.
            
        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens)]: Predicted bits.
        """
        T = 1 if self.pred == 0 else self.T 
        b, seq_len, *_ = embed.shape
        
        # create random noise
        crpt = torch.randn((b, seq_len, self.num_bits), device=embed.device)
        for t in range(T, 0, -T//self.steps):
            time = torch.full_like(crpt, fill_value=t, dtype=torch.float32, device=embed.device)
            crpt = self.denoise(crpt, time, embed, mask)
        return self.bit2tag(crpt)[mask]
            
    def progression(self, embed: torch.Tensor, mask: torch.Tensor) -> Iterable[torch.Tensor]:
        b, seq_len, *_ = embed.shape
        crpt = torch.randn((b, seq_len, self.num_bits), device=embed.device)
        for t in range(self.T, 0, -1):
            time = torch.full_like(crpt, fill_value=t, dtype=torch.float32, device=embed.device)
            crpt = self.denoise(crpt, time, embed, mask)
            yield self.bit2tag(crpt)[mask]
        
