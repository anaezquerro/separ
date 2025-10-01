from __future__ import annotations
import torch 

from separ.models.con.tetra.parser import TetraTaggingConstituencyParser
from separ.models.con.gat import ConstituencyGaTParser
from separ.data import PTB

class TetraTaggingConstituencyGaTParser(TetraTaggingConstituencyParser, ConstituencyGaTParser):
    NAME = 'con-gat-tetra'
    
    def transform(self, tree: PTB.Tree) -> PTB.Tree:
        if not tree.transformed:
            tree.TETRA, tree.FENCE, tree.CON, tree.LEAF = self.lab.encode(tree)
            tree.TETRA = ['<pad>']*len(tree) if len(tree.TETRA) == 0 else ['<pad>'] + tree.TETRA + ['<pad>']
            tree.FENCE = ['<pad>']*len(tree) if len(tree.FENCE) == 0 else ['<pad>'] + tree.FENCE + ['<pad>']
            tree.CON += ['<pad>']*(len(tree) - len(tree.CON))
            tree.transformed = True 
        return tree
    
    def _pred(self, tree: PTB.Tree, tetras: torch.Tensor, fences: torch.Tensor, cons: torch.Tensor, leaves: torch.Tensor) -> tuple[PTB.Tree, bool]:
        tetras, fences = tetras[1:-1], fences[1:-1]
        if len(tree) > 1:
            cons = cons[:-1]
        return TetraTaggingConstituencyParser._pred(self, tree, tetras, fences, cons, leaves)
    
