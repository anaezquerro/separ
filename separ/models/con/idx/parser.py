from __future__ import annotations
from argparse import ArgumentParser
from typing import Tuple, List, Dict, Union, Optional, Iterator
import torch, os 

from separ.parser import Parser
from separ.models.con.idx.model import IndexConstituencyModel
from separ.utils import ConstituencyMetric, create_mask, acc, avg, Config, flatten, parallel
from separ.data import PTB, AbstractTokenizer, Tokenizer, CharacterTokenizer, PretrainedTokenizer

def extra_leaves(tag: str) -> str:
    if PTB.Tree.UNARY not in tag:
        return ''
    else:
        *extra, _ = tag.split(PTB.Tree.UNARY)
        return PTB.Tree.UNARY.join(extra)

class IndexConstituencyParser(Parser):
    NAME = 'con-idx'
    MODEL = IndexConstituencyModel
    METRIC = ConstituencyMetric
    DATASET = PTB
    PARAMS = ['rel']
    
    class Labeler:
        def __init__(self, rel: bool = False):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexConstituencyLabeler(rel={self.rel})'
            
        def encode(self, tree: PTB.Tree) -> Tuple[List[str], List[str], List[str]]:
            collapsed = tree.collapse_unary()
            indexes = [0 for _ in range(len(tree))]
            lens = [len(tree)+1 for _ in range(len(tree))]
            cons = ['' for _ in range(len(tree))]
            for span in sorted(collapsed.spans, key=lambda s: (s.LEFT, len(s))):
                for i in range(span.LEFT, span.RIGHT-1):
                    indexes[i] += 1 
                    if lens[i] > len(span):
                        lens[i] = len(span)
                        cons[i] = span.LABEL
            leaves = list(map(extra_leaves, collapsed.POS))
            if self.rel:
                indexes = self.relativize(indexes)
            return list(map(str, indexes)), cons, leaves
    
        def relativize(self, indexes: List[Union[int, str]]) -> List[int]:
            return [int(indexes[0])] + [int(indexes[i]) - int(indexes[i-1]) for i in range(1, len(indexes))]
            
        def decode(self, indexes: List[str], cons: List[str], leaves: List[str]) -> List[PTB.Span]:
            stack, spans = [], []
            indexes = self.relativize(indexes) if not self.rel else list(map(int, indexes))
            for i, (index, con) in enumerate(zip(indexes, cons)):
                if index > 0: # open new spans (LEFT, RIGHT, LABEL)
                    stack += [(i, None, None) for _ in range(index-1)]
                    stack.append((i, None, con))
                elif index < 0: # close previous spans 
                    cnt, p = 0, len(stack)-1
                    while cnt < abs(index) and p >= 0:
                        if stack[p][-1] is not None:
                            cnt += 1 
                            span = stack.pop(p)
                            spans.append(PTB.Span(span[0], i+1, span[-1]))
                        p -= 1
                    p = len(stack) - 1
                    while p >= 0:
                        if stack[p][-1] is None or stack[p][-1] == con:
                            stack[p] = (stack[p][0], None, con)
                            break 
                        p -= 1
            return spans + [PTB.Span(i, i+1, tag) for i, tag in enumerate(leaves) if tag != '']
             
        def decode_postprocess(self, indexes: List[str], cons: List[str], leaves: List[str]) -> Tuple[List[PTB.Span], bool]:
            stack, spans = [], []
            indexes = self.relativize(indexes) if not self.rel else list(map(int, indexes))
            for i, (index, con) in enumerate(zip(indexes, cons)):
                if index > 0: # open new spans (LEFT, RIGHT, LABEL)
                    stack += [(i, None, None) for _ in range(index-1)]
                    stack.append((i, None, con))
                elif index < 0: # close previous spans 
                    cnt, p = 0, len(stack)-1
                    while cnt < abs(index) and p >= 0:
                        if len(stack) > 0 and stack[p][-1] is not None:
                            cnt += 1 
                            span = stack.pop(p)
                            spans.append(PTB.Span(span[0], i+1, span[-1]))
                        p -= 1
                    p = len(stack) - 1
                    while p >= 0:
                        if stack[p][-1] is None or stack[p][-1] == con:
                            stack[p] = (stack[p][0], None, con)
                            break 
                        p -= 1
            spans = spans + [PTB.Span(i, i+1, tag) for i, tag in enumerate(leaves) if tag != '']
            return spans, all(span.LABEL is not None for span in spans) and len(stack) == 0
        
        def test(self, tree: PTB.Tree) -> bool:
            labels = self.encode(tree)
            rec1 = PTB.Tree.from_spans(tree.preterminals, self.decode(*labels)).recover_unary()
            rec2, well_formed = self.decode_postprocess(*labels)
            rec2 = PTB.Tree.from_spans(tree.preterminals, rec2).recover_unary()
            return well_formed and tree == rec1 == rec2
        
    def __init__(
        self,
        model: IndexConstituencyModel,
        input_tkzs: List[AbstractTokenizer],
        target_tkzs: List[AbstractTokenizer],
        rel: bool,
        device: str
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.rel = rel 
        self.labeler = self.Labeler(rel)
        self.TRANSFORM_ARGS = [self.input_tkzs, *self.target_tkzs, self.labeler]
        
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--rel', action='store_true', help='Relative indexing')
        return argparser
    
    @classmethod
    def transform(
        cls, 
        tree: PTB.Tree, 
        input_tkzs: List[Tokenizer], 
        INDEX: Tokenizer, 
        CON: Tokenizer, 
        LEAF: Tokenizer, 
        labeler: IndexConstituencyParser.Labeler
    ):
        if not tree._transformed:
            for tkz in input_tkzs:
                tree.__setattr__(tkz.name, tkz.encode(getattr(tree, tkz.field)).pin_memory())
            indexes, cons, leaves = labeler.encode(tree)
            tree.INDEX = INDEX.encode(indexes).pin_memory()
            tree.CON = CON.encode(cons).pin_memory()
            tree.LEAF = LEAF.encode(leaves).pin_memory()
            tree._transformed = True 
    
    def collate(self, batch: List[PTB.Tree]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[PTB.Tree]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in batch]) for tkz in self.target_tkzs]
        masks = [create_mask(list(map(len, batch)))]
        return inputs, targets, masks, batch 
    
    def _pred(self, tree: PTB.Tree, index_pred: torch.Tensor, con_pred: torch.Tensor, tag_pred: torch.Tensor) -> Tuple[PTB.Tree, bool]:
        spans, well_formed = self.labeler.decode_postprocess(
            self.INDEX.decode(index_pred), 
            self.CON.decode(con_pred), 
            self.LEAF.decode(tag_pred)
        )
        return PTB.Tree.from_spans(tree.preterminals, spans).recover_unary(), well_formed
    
    def train_step(
        self, 
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        indexes, cons, leaves, mask = *targets, *masks
        s_index, s_con, s_tag = self.model(inputs[0], inputs[1:], mask)
        loss = self.model.loss(s_index, s_con, s_tag, indexes, cons, leaves)
        return loss, dict(iacc=acc(s_index, indexes), cacc=acc(s_con, cons), tacc=acc(s_tag, leaves))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        trees: List[PTB.Tree]
    ) -> Iterator[PTB.Tree]:
        lens = masks[0].sum(-1).tolist()
        index_preds, con_preds, tag_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, trees, index_preds.split(lens), con_preds.split(lens), tag_preds.split(lens)))
        return preds 
        
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        trees: List[PTB.Tree]
    ) -> Tuple[Dict[str, float], Iterator[PTB.Tree]]:
        """Returns debugging metrics with model predictions.

        Args:
            inputs (List[torch.Tensor]): Batched inputs.
            targets (List[torch.Tensor]): Batched targets.
            masks (List[torch.Tensor]): Padding mask.
            trees (List[PTB.Tree]): Input trees.

        Returns:
            Tuple[Dict[str, float], List[PTB.Tree]]: Debugging metrics and model predictions.
        """
        indexes, cons, leaves, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, index_preds, con_preds, leaf_preds = self.model.control(inputs[0], inputs[1:], *targets, *masks)
        preds, well_formed = zip(*map(self._pred, trees, index_preds.split(lens), con_preds.split(lens), leaf_preds.split(lens)))
        control = dict(loss=loss.item(), INDEX=acc(index_preds, indexes), CON=acc(con_preds, cons), LEAF=acc(leaf_preds, leaves), 
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
        rel: bool = False,
        pretest: bool = False, 
        device: str = 'cuda:0',
        num_workers: int = os.cpu_count(),
        **_
    ) -> IndexConstituencyParser:
        if isinstance(data, str):
            data = PTB.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            in_confs = [word_conf, None, None]
            word_conf.update(pad_index=input_tkzs[0].pad_index)
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
        index_tkz, con_tkz, tag_tkz = Tokenizer('INDEX'), Tokenizer('CON'), Tokenizer('LEAF')
        labeler = cls.Labeler(rel=rel)
        if pretest: 
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        indexes, cons, leaves = map(flatten, zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        index_tkz.train(*indexes)
        con_tkz.train(*cons)
        tag_tkz.train(*leaves)
        
        model = cls.MODEL(enc_conf, *in_confs, index_tkz.conf, con_tkz.conf, tag_tkz.conf).to(device)
        return cls(model, input_tkzs, [index_tkz, con_tkz, tag_tkz], rel, device)
            
        
            