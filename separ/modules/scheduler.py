from __future__ import annotations
import torch, pickle 
import plotly.graph_objects as go 
from typing import Callable


class NoiseScheduler:
    def __init__(self, T: int, beta: Callable):
        """Initialize noise scheduler for Stable Diffusion forward process.

        Args:
            T (int): Maximum number of time steps.
            beta (Callable): Noise function.
        """
        self.T = T 
        self.steps = torch.arange(T+1)
        self._beta = beta(self.steps)
        self._alpha = 1-self._beta 
        self._alpha_bar = torch.tensor([self._alpha[:t].prod() for t in range(1, len(self._alpha)+1)])
        self._sigma = torch.sqrt(self._beta)
        
    def __repr__(self) -> str:
        raise NotImplementedError
    
    def beta(self, time: torch.Tensor) -> torch.Tensor:
        time = time.clamp(min=0, max=self.T).long()
        return self._beta[time]
    
    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        time = time.clamp(min=0, max=self.T).long()
        return self._alpha[time]
    
    def alpha_bar(self, time: torch.Tensor) -> torch.Tensor:
        time = time.clamp(min=0, max=self.T).long()
        return self._alpha_bar[time]
    
    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        time = time.clamp(min=0, max=self.T).long()
        return self._sigma[time]

    def save(self, path: str):
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as reader:
            sched = pickle.load(reader)
        return sched 
    
    def to(self, device: str):
        self._beta = self._beta.to(device)
        self._alpha = self._alpha.to(device)
        self._sigma = self._sigma.to(device)
        self._alpha_bar = self._alpha_bar.to(device)
        self.steps = self.steps.to(device)
        return self

  
class ConstantScheduler(NoiseScheduler):
    def __init__(self, T: int, c: float = 5e-3):
        """Initialize a constant noise scheduler for Stable Diffusion forward process.
        
        Args:
            T (int): Maximum number of time steps.
            c (float): Constant value. Defaults to 0.005.
        """
        super().__init__(T, lambda t: torch.full_like(t, c))
        self.c = c

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(T={self.T}, c={self.c})'
        
class LinearScheduler(NoiseScheduler):
    def __init__(self, T: int, beta1: float = 1e-4, betaT: float = 0.02):
        """Initialize a linear noise scheduler for Stable Diffusion forward process.
        
        Args:
            T (int): Maximum number of time steps.
            beta1 (float): First noise level. Defaults to 0.0001.
            betaT (float): Last noise level. Defaults to 0.02.
        """
        a = beta1 - (beta1-betaT)/(1-T)
        b = (beta1-betaT)/(1-T)
        super().__init__(T, lambda t: (a+b*t).clamp(min=0))
        self.beta1 = beta1 
        self.betaT = betaT 
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(T={self.T}, beta1={self.beta1}, betaT={self.T})'
    
class CosineScheduler(NoiseScheduler):
    def __init__(self, T: int, s: float = 0.008):
        """Initialize a cosine noise scheduler for Stable Diffusion forward process.

        Args:
            T (int): Maximum number of time steps.
            s (float, optional): Scale value. Defaults to 0.008.
        """
        super().__init__(T, lambda t: torch.cos(((T-t)/T+s)/(1+s) * 0.5*torch.pi)**2)
        self.s = s 
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(T={self.T}, s={self.s})'