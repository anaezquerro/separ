from torch import nn 
from typing import Optional, Tuple 
import torch 
import plotly.graph_objects as go 

from separ.modules.scheduler import NoiseScheduler
from separ.modules.dit import DiT 

class BiT(nn.Module):
    """Bit transformer part of the Bit Diffusion Tagger from https://aclanthology.org/2023.findings-emnlp.860/."""
    
    def __init__(
        self, 
        hidden_size: int,
        sched: NoiseScheduler,
        num_bits: int, 
        max_len: int,
        num_layers: int,
        num_heads: int,
        pred: int,
        tol: float,
        patience: int,
        **_
    ): 
        """Initialization of the BiDiT model.

        Args:
            hidden_size (int): Dimension of the decoder embeddings.
            sched (NoiseScheduler): Noise scheduler.
            num_bits (int): Number of output bits per token.
            max_len (int): Maximum input length.
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of Transformer heads.
            pred (int): Type of prediction. `0` to directly predict the denoised signal, `-1` to 
                predict the previous latent and `1` to predict the noise.
            loss (int): Loss function (`1` for L1 and `1` for L2).
            tol (float): Inference tolerance.
            patience (int): Inference patience.
        """
        assert pred in [0, 1, -1], 'Prediction selection must be in {0, 1, -1}'
        super().__init__()
        self.sched = sched 
        self.T = self.sched.T 
        self.pred = pred 
        self.tol = tol 
        self.patience = patience 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_bits = num_bits
        self.decoder = DiT(self.num_bits, hidden_size, num_layers, max_len=max_len, num_heads=num_heads)
        
    def __repr__(self) -> str:
        return f'BiT(num_bits={self.num_bits}, T={self.T}, pred={self.pred}, num_layers={self.num_layers}, num_heads={self.num_heads})'
            
    def sample_time(self, batch_size: int) -> torch.Tensor:
        """Sample a random batch of discrete time steps.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Random batch of discrete time steps.
        """
        return (torch.rand(batch_size)*(self.T-1)+1).round().int()
    
    def corrupt(self, bits: torch.Tensor, time: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Corrupts a batch of bits by adding Gaussian noise.

        Args:
            bits (torch.Tensor ~ [batch_size, max(seq_lens), n_bits]): Batch of bits.
            time (torch.Tensor ~ batch_size): Batch of timesteps to corrupt each sequence.
            e (Optional[torch.Tensor], optional): Gaussian noise added to the forward process. If 
                None, no extra-noise is added, so the forward process is deterministic.

        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens), n_bits]: Batch of corrupted bits.
        """
        alpha_bar = self.sched.alpha_bar(time).view(bits.shape[0], 1, 1)
        crpt = torch.sqrt(alpha_bar)*bits 
        if e is not None:
            crpt += torch.sqrt(1-alpha_bar)*e 
        return crpt
    
    def forward(self, embed: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass.

        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Conditional contextualized embeddings.
            bits (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Tags to predict.
            mask (torch.Tensor): Padding mask.

        Returns:
            torch.Tensor ~ [sum(seq_lens), num_bits]: Model predictions.
            torch.Tensor ~ [sum(seq_lens), num_bits]: Model targets.
        """
        bits = (bits * 2 -1).to(torch.float32) 
        
        # corrupt bits 
        b = embed.shape[0]
        time = self.sample_time(b).to(embed.device)
        e = torch.rand_like(bits)
        crpt = self.corrupt(bits, time, e)
        preds = self.decoder(crpt, time, embed, mask)
        
        if self.pred == 1: # noise as target
            return preds[mask], e[mask]
        elif self.pred == -1: # previous latent as target
            time = (time-1).clamp(min=0)
            prev = torch.where(time.view(b, 1, 1) == 0, bits, self.corrupt(bits, time, torch.randn_like(bits)))
            return preds[mask], prev[mask]
        else: # denoised signal as target
            return preds[mask], bits[mask]
        
    def denoise(self, crpt: torch.Tensor, time: torch.Tensor, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Performs one denoising step.
        
        Args:
            crpt (torch.Tensor ~ [batch_size, max(seq_lens), n_bits]): Corrupted bits.
            time (torch.Tensor ~ batch_size): Time step of the forward process.
            embed (torch.Tensor ~ [batch_size, max(seq_lens), embed_dim]): Conditional embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.

        Returns:
            torch.Tensor: Previous latent.
        """
        b = time.shape[0]
        time = time.clamp(max=self.T)
        preds = self.decoder(crpt, time, embed, mask)

        if self.pred == 1:
            alpha_bar = self.sched.alpha_bar(time).view(b, 1, 1)
            # alpha = self.sched.alpha(time).view(b, 1, 1)
            sigma = self.sched.sigma(time).view(b, 1, 1)
            # prev = (1/torch.sqrt(alpha))*(crpt - (1-alpha)/torch.sqrt(1-alpha_bar)*preds)
            prev = crpt - preds*torch.sqrt(1-alpha_bar)
            prev = torch.where((time-1).clamp(min=0).view(b, 1, 1) == 0, prev, prev + sigma * torch.rand_like(prev))
        else:
            prev = preds 
            sigma = self.sched.sigma(time).view(b, 1, 1)
            prev = torch.where(time.view(b, 1, 1) == 0, prev, prev + sigma * torch.rand_like(prev))
        return torch.where(time.view(b, 1, 1) > 0, prev, crpt)
    
    def predict(self, embed: torch.Tensor, mask: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Prediction step.
        
        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), fix_len]): Conditional contextualized embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.
            steps (Optional[int]): Number of inference steps.
            
        Returns:
            torch.Tensor ~ [sum(seq_lens), num_bits]: Predicted bits.
        """
        steps = 1 if self.pred == 0 else (steps or self.T)
        b, seq_len, *_ = embed.shape
        
        # create random noise
        crpt = torch.randn((b, seq_len, self.num_bits), device=embed.device)
        for t in range(steps, 0, -1):
            time = torch.full(size=(b,), fill_value=t, dtype=torch.float32, device=embed.device)
            crpt = self.denoise(crpt, time, embed, mask)
        return crpt[mask]
            
    def validate(
        self, 
        embed: torch.Tensor, 
        bits: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Validation step to find the optimal number of timesteps.
        
        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Conditional contextualized embeddings.
            bits (torch.Tensor ~ [batch_size, max(seq_lens), num_bits]): Tags to predict.
            mask (torch.Tensor): Padding mask.
            tol (float): Tolerance, minimum improvement at each step.
            patience (int): Number of steps allowed with no improvement.
        
        Returns:  
            Tuple[torch.Tensor ~ [sum(seq_lens), num_bits], int]: Predictions and number of steps.
        """
        b, seq_len, *_ = embed.shape
        crpt = torch.randn((b, seq_len, self.num_bits), device=embed.device)
        T = 1 if self.pred == 0 else self.T
        best, best_t, improv = crpt, T, self.patience 
        best_score = ((crpt[mask] > 0).int() == (bits[mask] > 0).int()).float().mean()
        
        for t in range(T, 0, -1):
            time = torch.full((b,), t, dtype=torch.float32, device=embed.device)
            crpt = self.denoise(crpt, time, embed, mask)
            score = ((crpt[mask] > 0).int() == (bits[mask] > 0).int()).float().mean()
            
            if score - best_score > self.tol: # it has improved more than tol 
                best = crpt 
                best_t = t 
                improv = self.patience
                best_score = score
            else:
                improv -= 1
                
            if improv == 0:
                break 
        return best[mask], best_t 

        
            
    
    
        

        
