from __future__ import annotations
from typing import List, Tuple, Union, Optional
from argparse import ArgumentParser
import torch

from separ.models.con.parser import ConstituencySLParser
from separ.utils import Config, flatten, bar
from separ.data import PTB, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer

class TetraTaggingConstituencyParser(ConstituencySLParser):
    NAME = 'con-tetra'
    
    class Labeler:
        ROOT = '$'
        
        def __init__(self, binarize: int = 1):
            self.binarize = binarize 
        
        def __repr__(self) -> str:
            return f'TetraTaggingConstituencyLabeler(binarize={self.binarize})'
        
        def encode(self, tree: PTB.Tree) -> Tuple[List[str], List[str], List[str], List[str]]:
            """Tetra-tagging encoding.

            Args:
                tree (PTB.Tree): Input constituency tree.

            Returns:
                List[str] ~ (n-2): Token tags (first and last token are always > and <, respectively).
                List[str] ~ (n-2): Fencepost tags (first token is always >).
                List[str] ~ (n-1): Fencepost constituents.
                List[str] ~ n: Pre-terminal nodes.
            """
            btree = tree.binarize(self.binarize)
            if len(btree) == 1:
                return [], [], [btree.label], btree.POS
            tetras, fences, cons = self.tetra(btree)
            return tetras[1:-1], fences[1:], cons, btree.POS
                    
        def tetra(self, tree: PTB.Tree, tetras: List[str] = [], fences: List[str] = ['>']) -> List[str]:
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
            tetras: List[str],
            fences: List[str],
            cons: List[str], 
            leaves: List[str],
            words: List[str]
        ) -> Tuple[PTB.Tree, bool]:
            stack, well_formed = [], True
            if len(leaves) == 1:
                leaf = PTB.Tree.from_leaf(leaves[0], words[0])
                if cons[0] == leaves[0]:
                    return leaf.recover_unary(), True 
                else:
                    return PTB.Tree(cons[0], deps=[leaf]).recover_unary(), True
            
            for tetra, fence, con in zip(['>'] + tetras, ['<'] + fences, cons):
                if (tetra == '>' or len(stack) < 1) and (sum(s.label == self.ROOT for s in stack) < len(words)):
                    stack.append(PTB.Tree(self.ROOT, deps=[PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))]))
                    well_formed &= (tetra == '>')
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
                    stack.pop(-1)
                    well_formed &= (tetra == '<')
                    
                if (fence == '>' or len(stack) < 2) and (sum(s.label == self.ROOT for s in stack) < len(words) + 1):
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                    well_formed &= (fence == '>')
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
                    well_formed &= (fence == '<')
            stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
            return stack[0].deps[0].debinarize(), well_formed
        
        def test(self, tree: PTB.Tree) -> bool:
            rec, well_formed = self.decode(*self.encode(tree), tree.FORM)
            return rec == tree and well_formed
        
    def __init__(
        self,
        input_tkzs: List[InputTokenizer], 
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.lab = self.Labeler()
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(TetraTaggingConstituencyParser, cls).add_arguments(argparser)
        return argparser 
        
    def transform(self, tree: PTB.Tree) -> PTB.Tree:
        if not tree.transformed:
            tree.TETRA, tree.FENCE, tree.CON, tree.LEAF = self.lab.encode(tree)
            tree.transformed = True 
        return tree 
    
    def collate(self, trees: List[PTB.Tree]):
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
        
    def _pred(self, tree: PTB.Tree, *preds: List[torch.Tensor]) -> Tuple[PTB.Tree, bool]:
        rec, well_formed = self.lab.decode(*[tkz.decode(pred) for pred, tkz in zip(preds, self.target_tkzs)], tree.FORM)
        rec.ID = tree.ID
        return rec, well_formed
  
    @classmethod
    def build(
        cls, 
        data: Union[PTB, str],
        enc_conf: Config,
        word_conf: Config,
        char_conf: Optional[Config] = None,
        device: int = 0,
        **_
    ) -> TetraTaggingConstituencyParser:
        if isinstance(data, str):
            data = PTB.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            word_tkz = InputTokenizer('WORD', 'FORM')
            word_tkz.train(data)
            word_conf.update(word_tkz.conf)
            input_tkzs = [word_tkz]
            
            if char_conf is not None:
                char_tkz = CharacterTokenizer('CHAR', 'FORM')
                char_tkz.train(data)
                char_conf.update(char_tkz.conf)
                input_tkzs.append(char_tkz)
                
            in_confs = [word_conf, None, char_conf]
        
        # train target tokenizers 
        tetra_tkz = TargetTokenizer('TETRA')
        fence_tkz = TargetTokenizer('FENCE')
        con_tkz = TargetTokenizer('CON')
        leaf_tkz = TargetTokenizer('LEAF')
        labeler = cls.Labeler()
        labels, fences, cons, leaves = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        tetra_tkz.train(labels)
        fence_tkz.train(fences)
        con_tkz.train(cons)
        leaf_tkz.train(leaves)
        
        return cls(input_tkzs, [tetra_tkz, fence_tkz, con_tkz, leaf_tkz], 
                   [enc_conf, *in_confs, tetra_tkz.conf, fence_tkz.conf, con_tkz.conf, leaf_tkz.conf], 
                   device)
            
        
            