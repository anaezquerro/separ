from __future__ import annotations
from argparse import ArgumentParser
import torch

from separ.models.con.parser import ConstituencySLParser
from separ.utils import Config, flatten, bar
from separ.data import PTB, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer

class TetraTaggingConstituencyParser(ConstituencySLParser):
    NAME = 'con-tetra'
    
    class Labeler:
        ROOT = '$'
        PARAMS = {'binarize'}
        NAME = 'con-tetra'
        NAMES = ['TETRA', 'FENCE', 'CON', 'POS']
        
        def __init__(self, binarize: int, **_):
            self.binarize = binarize 
        
        def __repr__(self) -> str:
            return f'TetraTaggingConstituencyLabeler(binarize={self.binarize})'
        
        def encode(self, tree: PTB.Tree) -> tuple[list[str], list[str], list[str], list[str]]:
            """Tetra-tagging encoding.

            Args:
                tree (PTB.Tree): Input constituency tree.

            Returns:
                list[str] ~ (n-2): Token tags (first and last token are always > and <, respectively).
                list[str] ~ (n-2): Fencepost tags (first token is always >).
                list[str] ~ (n-1): Fencepost constituents.
                list[str] ~ n: Pre-terminal nodes.
            """
            btree = tree.binarize(self.binarize)
            if len(btree) == 1:
                return [], [], [btree.label], btree.POS
            tetras, fences, cons = self.tetra(btree)
            return tetras[1:-1], fences[1:], cons, btree.POS
                    
        def tetra(self, tree: PTB.Tree, tetras: list[str] = [], fences: list[str] = ['>']) -> list[str]:
            left, right = tree.deps 
            if left.is_preterminal():
                left_tetras, left_fences, left_cons = ['>'], [],  []
            else:
                left_tetras, left_fences, left_cons = self.tetra(left, [], ['>'])
            if right.is_preterminal():
                right_tetras, right_fences, right_cons = ['<'], [], []
            else:
                right_tetras, right_fences, right_cons = self.tetra(right, [], ['<'])
            return left_tetras + tetras + right_tetras, left_fences + fences + right_fences, left_cons + [tree.label] + right_cons

        def decode(
            self, 
            tetras: list[str],
            fences: list[str],
            cons: list[str], 
            leaves: list[str],
            words: list[str]
        ) -> tuple[PTB.Tree, bool]:
            stack, well_formed = [], True
            if len(leaves) == 1:
                leaf = PTB.Tree.from_leaf(leaves[0], words[0])
                if cons[0] == leaves[0]:
                    return leaf.recover_unary(), True 
                else:
                    return PTB.Tree(cons[0], deps=[leaf]).recover_unary(), True
            
            for tetra, fence, con in zip(['>'] + tetras, ['>'] + fences, cons):
                well_formed &= (tetra == '<' and len(stack) >= 1) or (tetra == '>')
                if (tetra == '>' or len(stack) < 1) and (sum(s.label == self.ROOT for s in stack) < len(words)):
                    stack.append(PTB.Tree(self.ROOT, deps=[PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))]))
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
                    stack.pop(-1)
                    
                well_formed &= (fence == '<' and len(stack) >= 2) or (fence == '>')
                if (fence == '>' or len(stack) < 2) and (sum(s.label == self.ROOT for s in stack) < len(words) + 1):
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
                    
            stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
            return stack[0].deps[0].debinarize(), well_formed
        
        def test(self, tree: PTB.Tree) -> bool:
            rec, well_formed = self.decode(*self.encode(tree), tree.FORM)
            return rec == tree and well_formed
        
    def collate(self, trees: list[PTB.Tree]):
        inputs, (tmask, fmask, cmask, lmask), targets, *_ = super().collate(trees)
        lens = lmask.sum(-1)
        
        tmask[:, 0] = False 
        fmask[:, 0] = False 
        
        for i, l in enumerate(lens):
            tmask[i, l-1] = False 
            fmask[i, l-1] = False 
            if l > 1:
                cmask[i, l-1] = False 
        
        return inputs, [tmask, fmask, cmask, lmask], targets, trees
        
    def _pred(self, tree: PTB.Tree, *preds: list[torch.Tensor]) -> tuple[PTB.Tree, bool]:
        rec, well_formed = self.lab.decode(*[tkz.decode(pred) for pred, tkz in zip(preds, self.target_tkzs)], tree.FORM)
        rec.ID = tree.ID
        return rec, well_formed
  