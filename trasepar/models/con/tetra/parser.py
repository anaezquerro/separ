from __future__ import annotations
from typing import List, Tuple, Union, Dict, Iterator, Optional
import torch, os

from trasepar.parser import Parser 
from trasepar.models.con.tetra.model import TetraTaggingConstituencyModel
from trasepar.utils import ConstituencyMetric, create_mask, acc, avg, Config, flatten, parallel
from trasepar.data import PTB, AbstractTokenizer, Tokenizer, PretrainedTokenizer, CharacterTokenizer


class TetraTaggingConstituencyParser(Parser):
    NAME = 'con-tetra'
    MODEL = TetraTaggingConstituencyModel
    METRIC = ConstituencyMetric
    DATASET = PTB
    PARAMS = []
    
    class Labeler:
        ROOT = '$'
        
        def __init__(self):
            pass 
        
        def __repr__(self) -> str:
            return f'TetraTaggingConstituencyLabeler'
        
        def encode(self, tree: PTB.Tree) -> Tuple[List[str], List[str], List[str], List[str]]:
            btree = tree.binarize()
            if len(btree) == 1:
                return ['>'], ['>'], [btree.label], btree.POS
            tetras, fences, cons = self.tetra(btree)
            return tetras[:-1], fences, cons, btree.POS
                    
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

        def decode(self, tetras: List[str], fences: List[str], cons: List[str], leaves: List[str], words: List[str]) -> PTB.Tree:
            stack = []
            if len(leaves) == 1:
                return PTB.Tree(cons[0], PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))).recover_unary()
            for tag, fence, con in zip(tetras, fences, cons):
                if tag == '>':
                    stack.append(PTB.Tree('$', deps=[PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))]))
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
                    stack.pop(-1)
                if fence == '>':
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
            stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
            return stack[0].deps[0].debinarize()
        
        def decode_postprocess(
                self, 
                tetras: List[str],
                fences: List[str],
                cons: List[str], 
                leaves: List[str],
                words: List[str]
            ) -> Tuple[PTB.Tree, bool]:
            stack, well_formed = [], True
            if len(leaves) == 1:
                return PTB.Tree(cons[0], PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))).recover_unary(), True
            for tag, fence, con in zip(tetras, fences, cons):
                if tag == '>' or len(stack) <= 1:
                    stack.append(PTB.Tree('$', deps=[PTB.Tree.from_leaf(leaves.pop(0), words.pop(0))]))
                    well_formed = tag == '>'
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
                    stack.pop(-1)
                if fence == '>' or len(stack) <= 2:
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                    well_formed = fence == '>'
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
            stack[-1].deps.append(PTB.Tree.from_leaf(leaves.pop(0), words.pop(0)))
            if len(stack) == 1:
                tree = stack[0].deps[0].debinarize()
            else:
                stack = [t.deps[0] for t in stack if t.label == '$']
                while len(stack) > 1:
                    last = stack.pop(-1)
                    stack[-1].deps.append(last)
                tree = stack[0].debinarize()
                well_formed = False
            return tree, well_formed
        
        def test(self, tree: PTB.Tree) -> bool:
            tetras, fences, cons, leaves = self.encode(tree)
            rec1 = self.decode(tetras, fences, cons, leaves.copy(), tree.FORM)
            rec2, well_formed = self.decode_postprocess(tetras, fences, cons, leaves, tree.FORM)
            return rec1 == rec2 == tree and well_formed
        
    def __init__(
        self,
        model: TetraTaggingConstituencyModel, 
        input_tkzs: List[AbstractTokenizer], 
        target_tkzs: List[AbstractTokenizer],
        device: Union[str, int]):
        super().__init__(model, input_tkzs, target_tkzs, device)    
        self.labeler = self.Labeler()
        self.TRANSFORM_ARGS = [self.input_tkzs, *self.target_tkzs, self.labeler]
        
    @classmethod
    def transform(
        cls,
        tree: PTB.Tree,
        input_tkzs: List[Tokenizer],
        TETRA: Tokenizer,
        FENCE: Tokenizer,
        CON: Tokenizer,
        LEAF: Tokenizer,
        labeler: TetraTaggingConstituencyParser.Labeler
    ):
        if not tree._transformed:
            for tkz in input_tkzs:
                tree.__setattr__(tkz.name, tkz.encode(getattr(tree, tkz.field)).pin_memory())
            tetras, fences, cons, leaves = labeler.encode(tree)
            tree.TETRA = TETRA.encode(tetras).pin_memory()
            tree.FENCE = FENCE.encode(fences).pin_memory()
            tree.CON = CON.encode(cons).pin_memory()
            tree.LEAF = LEAF.encode(leaves).pin_memory()
            tree._transformed = True 
            
    def collate(self, batch: List[PTB.Tree]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[PTB.Tree]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in batch]) for tkz in self.target_tkzs]
        lens = list(map(len, batch))
        mask0 = create_mask(lens)
        mask1 = mask0.clone()
        for i, l in enumerate(lens):
            mask1[i, max(l-1, 1):] = False 
        return inputs, targets, [mask0, mask1], batch 
            
    def _pred(self, tree: PTB.Tree, tetra_pred: torch.Tensor, fence_pred: torch.Tensor, con_pred: torch.Tensor, leaf_pred: torch.Tensor) -> Tuple[PTB.Tree, bool]:
        rec, well_formed = self.labeler.decode_postprocess(
            self.TETRA.decode(tetra_pred), 
            self.FENCE.decode(fence_pred),
            self.CON.decode(con_pred), 
            self.LEAF.decode(leaf_pred),
            tree.FORM,
        )
        rec.ID = tree.ID
        return rec, well_formed
    
    def train_step(
        self, 
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        tetras, fences, cons, leaves, mask0, mask1 = *targets, *masks
        s_tetra, s_fence, s_con, s_leaf = self.model(inputs[0], inputs[1:], mask0, mask1)
        loss = self.model.loss(s_tetra, s_fence, s_con, s_leaf, *targets)
        return loss, dict(TETRA=acc(s_tetra, tetras), FENCE=acc(s_fence, fences), CON=acc(s_con, cons), LEAF=acc(s_leaf, leaves))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        trees: List[PTB.Tree]
    ) -> Iterator[PTB.Tree]:
        lens0, lens1 = masks[0].sum(-1).tolist(), masks[1].sum(-1).tolist()
        tetra_preds, fence_preds, con_preds, leaf_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, trees, tetra_preds.split(lens1), fence_preds.split(lens1), 
                            con_preds.split(lens1), leaf_preds.split(lens0)))
        return preds 
        
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        trees: List[PTB.Tree]
    ) -> Tuple[Dict[str, float], Iterator[PTB.Tree]]:
        tetras, fences, cons, leaves, mask0, mask1 = *targets, *masks
        lens0, lens1 = mask0.sum(-1).tolist(), mask1.sum(-1).tolist()
        loss, tetra_preds, fence_preds, con_preds, leaf_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        preds, well_formed = zip(*map(self._pred, trees, tetra_preds.split(lens1),
                                      fence_preds.split(lens1), con_preds.split(lens1), leaf_preds.split(lens0)))
        control = dict(loss=loss.item(), 
                       TETRA=acc(tetra_preds, tetras), 
                       FENCE=acc(fence_preds, fences),
                       CON=acc(con_preds, cons), 
                       LEAF=acc(leaf_preds, leaves), 
                       well_formed=avg(well_formed)*100)
        return control, preds
    
    
    @classmethod
    def build(
        cls, 
        data: Union[PTB, str],
        enc_conf: Config,
        word_conf: Config,
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        pretest: bool = False, 
        device: str = 'cuda:0',
        num_workers: int = os.cpu_count(),
        **_
    ) -> TetraTaggingConstituencyParser:
        if isinstance(data, str):
            data = PTB.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            word_conf |= input_tkzs[0].conf
            in_confs = [word_conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'POS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
            
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.update(vocab_size=len(tkz), pad_index=tkz.pad_index)
        
        # train target tokenizers 
        tetra_tkz, fence_tkz, con_tkz, leaf_tkz = Tokenizer('TETRA'), Tokenizer('FENCE'), Tokenizer('CON'), Tokenizer('LEAF')
        labeler = cls.Labeler()
        if pretest: 
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        tetras, fences, cons, leaves = map(flatten, zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        tetra_tkz.train(*tetras)
        fence_tkz.train(*fences)
        con_tkz.train(*cons)
        leaf_tkz.train(*leaves)
        
        model = cls.MODEL(enc_conf, *in_confs, tetra_tkz.conf, fence_tkz.conf, con_tkz.conf, leaf_tkz.conf).to(device)
        return cls(model, input_tkzs, [tetra_tkz, fence_tkz, con_tkz, leaf_tkz], device)
            
        
            