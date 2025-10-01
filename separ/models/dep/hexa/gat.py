from __future__ import annotations
import torch 

from separ.models.dep.hexa.parser import HexaTaggingDependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL

class HexaTaggingDependencyGaTParser(HexaTaggingDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-hexa'

    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.HEXA, tree.FENCE, tree.REL = self.lab.encode(tree)
            tree.HEXA = ['<pad>']*len(tree) if len(tree.HEXA) == 0 else ['<pad>'] + tree.HEXA + ['<pad>']
            tree.FENCE += ['<pad>']*(len(tree) - len(tree.FENCE))
            tree.transformed = True 
        return tree
    
    def _pred(self, tree: CoNLL.Tree, hexas: torch.Tensor, fences: torch.Tensor, rels: torch.Tensor) -> tuple[CoNLL.Tree, bool]:
        return HexaTaggingDependencyParser._pred(self, tree, hexas[1:-1], fences[:-1], rels)
    