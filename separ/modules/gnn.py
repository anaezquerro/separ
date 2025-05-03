from torch import nn 
from separ.modules.ffn import FFN
import torch 
from typing import Optional, Tuple

class GraphNeuralLayer(nn.Module):
    """Simplest implementation of a Graph Neural Layer (https://distill.pub/2021/gnn-intro/) with 
    node-level features."""
    
    def __init__(
        self,
        input_size: int, 
        output_size: int,
        pred_global: bool = False
    ):
        super().__init__()
        self.neighbor = FFN(input_size, output_size)
        self.node = FFN(input_size, output_size)
        self.glob = FFN(input_size, output_size) if pred_global else nn.Identity()
        self.input_size, self.output_size = input_size, output_size

    def __repr__(self) -> str:
        return f'GraphNeuralLayer(input_size={self.input_size}, output_size={self.output_size})'
        
    def forward(self, nodes: torch.Tensor, adjacent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass of the Graph Neural Layer (pooling operator).

        Args:
            nodes (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Node embeddings.
            adjacent (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Adjacent matrix (b, out, in).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Update node-level and global features.
        """
        # pool operation between neighbors
        b, out, _ = adjacent.nonzero(as_tuple=True)
        bins, steps = torch.unique(adjacent.nonzero()[:, [0,2]], dim=0, return_counts=True)
        out_nodes = nodes[b, out].split(steps.tolist())
        updated = self.node(nodes)
        for (b, inn), pool in zip(bins.tolist(), out_nodes):
            updated[b, inn] = self.neighbor(pool.sum(0))
        
        # compute global embedding 
        glob = self.glob(nodes.sum(1))
        return updated, glob
    
class GNN(nn.Module):
    def __init__(
        self, 
        n_layers: int, 
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        pred_global: bool = False
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        
        self.layers = nn.ModuleList([GraphNeuralLayer(input_size, hidden_size, pred_global=False)])
        for _ in range(n_layers-2):
            self.layers.append(GraphNeuralLayer(hidden_size, hidden_size, pred_global=False))
        self.layers.append(GraphNeuralLayer(hidden_size, self.output_size, pred_global=pred_global))
        
    def __repr__(self) -> str:
        return f'GNN(n_layers={self.n_layers}, input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})'
    
    def forward(self, nodes: torch.Tensor, adjacent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            nodes, glob = layer(nodes, adjacent)
        return nodes, glob
    
class GraphConvolutionalLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, act: nn.Module = nn.LeakyReLU(0.1)):
        super().__init__()
        self.W = nn.Parameter(data=torch.zeros(input_size, output_size), requires_grad=True)
        self.B = nn.Parameter(data=torch.zeros(input_size, output_size), requires_grad=True)
        self.act = act 
        self.reset_parameters()
        
    def forward(
        self,
        nodes: torch.Tensor,
        adjacent: torch.Tensor
    ) -> torch.Tensor:
        """Forward-pass of the Graph Neural Layer (pooling operator).

        Args:
            nodes (torch.Tensor ~ [batch_size, max(seq_lens), input_size]): Node embeddings.
            adjacent (torch.Tensor ~ [batch_size, max(seq_lens), max(seq_lens)]): Adjacent matrix (b, out, in).
            
        Returns:
            torch.Tensor: Update node-level features.
        """
        b, out, _ = adjacent.nonzero(as_tuple=True)
        bins, steps = torch.unique(adjacent.nonzero()[:, [0,2]], dim=0, return_counts=True)
        out_nodes = nodes[b, out].split(steps.tolist())
        updated = nodes @ self.B
        for (b, inn), pool in zip(bins.tolist(), out_nodes):
            updated[b, inn] += (pool.mean(0) @ self.W)
        return self.act(updated)
        
    def reset_parameters(self):
        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.B)
        
        
class GCN(nn.Module):
    def __init__(self, n_layers: int, input_size: int, hidden_size: int, output_size: Optional[int] = None):
        super().__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        
        self.layers = nn.ModuleList([GraphConvolutionalLayer(input_size, hidden_size)])
        for _ in range(n_layers-2):
            self.layers.append(GraphConvolutionalLayer(hidden_size, hidden_size))
        self.layers.append(GraphConvolutionalLayer(hidden_size, self.output_size))
        
    def forward(self, nodes: torch.Tensor, adjacent: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            nodes = layer(nodes, adjacent)
        return nodes 
            
            