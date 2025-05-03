from typing import List 
import torch 

from separ.structs.graph import cycles

def MST(scores: torch.Tensor) -> torch.Tensor:
    """Maximum-Spanning Tree algorithm from [Jurafsky (2019)](https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf).

    Args:
        scores (torch.Tensor): Input score matrix.

    Returns:
        torch.Tensor: Adjacent matrix of a dependency tree.
    """
    scores = preprocess(scores)
    n = scores.shape[0]
    heads = scores.argmax(-1)
    adjacent = torch.zeros(n, n, dtype=torch.bool, device=scores.device)
    adjacent[list(range(n)), heads] = True 
    
    for cycle in cycles(adjacent):
        if cycle != [0]:
            scores = contract(scores - scores.max(-1).values.unsqueeze(-1), sorted(cycle))
            new = MST(scores[:, :, 0])
            adjacent = expand(scores, new, adjacent)
            break 
    adjacent[0, :] = False
    return adjacent

def contract(scores: torch.Tensor, cycle: List[int]) -> torch.Tensor:
    """Contract operation as described in https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf.

    Args:
        scores (torch.Tensor): Matrix of scores.
        cycle (List[int]): Nodes of the detected cycle.

    Returns:
        torch.Tensor: Contract indexed score matrix of the reduced graph.
    """
    n = scores.shape[0]
    indices = torch.arange(n, device=scores.device).unsqueeze(-1).repeat(1,n)
    scores = torch.stack([scores, indices, indices.T], -1)
    pivot = min(cycle)
    
    # update rows 
    for head in range(n):
        view = scores[cycle, head, 0]
        view = torch.where(view == 0, view.min(), view)
        dep = cycle[view.argmax()]
        scores[pivot, head] = scores[dep, head]
        
    # update cols
    for dep in range(n):
        view = scores[dep, cycle, 0]
        view = torch.where(view == 0, view.min(), view)
        head = cycle[view.argmax()]
        scores[dep, pivot] = scores[dep, head]
        
    cycle.remove(pivot)
    maintain = [i for i in range(n) if i not in cycle]
    return scores[maintain][:, maintain]

def expand(scores: torch.Tensor, adjacent: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Expand operation as described in https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf.

    Args:
        scores (torch.Tensor ~ [seq_len, seq_len, 3]): Indexed matrix of scores (score, dep, head).
        adjacent (torch.Tensor): Estimated adjacent matrix.
        prev (torch.Tensor): Previous expanded adjacent matrix.

    Returns:
        torch.Tensor: New expanded matrix.
    """
    expanded = torch.zeros_like(prev, dtype=torch.bool)
    for idep, ihead in adjacent.nonzero():
        _, dep, head = scores[idep, ihead]
        expanded[dep.int().item(), head.int().item()] = True 
    for dep in range(prev.shape[0]):
        if expanded[dep].sum() == 0:
            expanded[dep] = prev[dep]
    return expanded

def preprocess(scores: torch.Tensor) -> torch.Tensor:
    """Preprocess a scored adjacent matrix to fulfill the dependency-tree constraints.

    Args:
        scores (torch.Tensor): Scores of the adjacent matrix.

    Returns:
        torch.Tensor: Preprocessed matrix.
    """
    n = scores.shape[0]
    # suppress diagonal 
    scores[list(range(n)), list(range(n))] = scores.min()-1
    
    # only one root
    root = scores[:, 0].argmax(-1)
    scores[:, 0] = scores.min()-1
    scores[root, 0] = scores.max()+1
    
    # cycle in w0 -> w0
    scores[0, 0] = scores.max()
    return scores 