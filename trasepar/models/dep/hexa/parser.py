from __future__ import annotations
from typing import List, Union, Optional, Tuple, Iterator, Dict
from argparse import ArgumentParser
import torch 

from trasepar.parser import Parser 
from trasepar.models.dep.hexa.model import HexaTaggingDependencyModel
from trasepar.data import CoNLL, Tokenizer, PretrainedTokenizer, CharacterTokenizer, PTB
from trasepar.utils import DependencyMetric, Config, create_mask, acc, flatten, parallel, avg
from trasepar.structs import Arc 
from trasepar.models.dep.labeler import DependencyLabeler

class HexaTaggingDependencyParser(Parser):
    NAME = 'dep-hexa'
    MODEL = HexaTaggingDependencyModel
    DATASET = CoNLL
    METRIC = DependencyMetric
    PARAMS = ['proj']
    
    class Labeler(DependencyLabeler):
        LEFT = '<L>'
        RIGHT = '<R>'
        ROOT = '$'
        
        def __init__(self, proj: Optional[str]):
            self.proj = proj 
        
        def __repr__(self) -> str:
            return f'HexaTaggingDependencyLabeler(proj={self.proj})'
        
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str], List[str]]:
            if self.proj:
                graph = graph.projectivize(self.proj)
            if len(graph) == 1:
                return ['>'], ['>'], [self.LEFT]

            # Step 1: Transform dependency graph into BHT 
            stack = []
            self.dep2tree(0, graph, stack)
            tree = stack[0][1]
            
            # Step 2: Obtain labels of tree
            tags, fences, cons = self.encode_bht(tree, [], ['>'])
            return tags[:-1], fences, cons
        
        def decode(self, tags: List[str], fences: List[str], cons: List[str], rels: List[str]) -> List[Arc]:
            if len(rels) == 1:
                return [Arc(0, 1, rels[0])]
            # Step 1: Obtain the BHT 
            tree = self.decode_bht(tags, fences, cons, len(rels))
            # Step 2: Recover the dependency tree
            arcs = []
            self.tree2dep(PTB.Tree(self.ROOT, [tree]), arcs)
            for arc, rel in zip(sorted(arcs), rels):
                arc.REL = rel 
            return arcs
        
        def decode_postprocess(self, tags: List[str], fences: List[str], cons: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            if len(rels) == 1:
                return [Arc(0, 1, rels[0])], True 
            # Step 1: Obtain the BHT 
            tree, well_formed = self.decode_postprocess_bht(tags, fences, cons, len(rels))

            # Step 2: Recover the dependency tree
            arcs = []
            pos = list(map(str, range(1, len(rels)+1)))
            btree = tree.binarize().rebuild_terminals(pos)
            self.tree2dep(PTB.Tree(self.ROOT, [btree]), arcs)
            for arc, rel in zip(sorted(arcs), rels):
                arc.REL = rel 
            return arcs, well_formed

        def dep2tree(self, head: int, graph: CoNLL.Graph, stack: List[int]):
            stack.append(head)
            # get left and right dependencies
            lefts, rights = [], []
            for arc in graph.arcs:
                if arc.HEAD == head:
                    if arc.DEP < arc.HEAD:
                        lefts.append(arc.DEP)
                    else:
                        rights.append(arc.DEP)
            for dep in sorted(lefts, reverse=True):
                self.dep2tree(dep, graph, stack)
                left = stack.pop(-1)
                right = stack.pop(-1)
                if isinstance(left, int):
                    left = PTB.Tree.from_leaf(str(left), graph.FORM[left-1])
                if isinstance(right, int):
                    right = PTB.Tree.from_leaf(str(right), graph.FORM[right-1])
                stack.append(PTB.Tree(self.RIGHT, deps=[left, right]))
            for dep in rights:
                self.dep2tree(dep, graph, stack)
                right = stack.pop(-1)
                left = stack.pop(-1)
                if isinstance(left, int):
                    left = PTB.Tree.from_leaf(str(left), graph.FORM[left-1])
                if isinstance(right, int):
                    right = PTB.Tree.from_leaf(str(right), graph.FORM[right-1])
                stack.append(PTB.Tree(self.LEFT, deps=[left, right]))
                
        def tree2dep(self, node: PTB.Tree, arcs: List[Arc]):
            if node.is_preterminal(): 
                return node.deps[0]
            elif node.is_terminal():
                return node
            left = self.tree2dep(node.deps[0], arcs)
            if len(node.deps) == 1:
                arcs.append(Arc(0, int(left.label), None))
                return 
            else:
                right = self.tree2dep(node.deps[1], arcs)
            if node.label == self.LEFT:
                arcs.append(Arc(int(left.label), int(right.label), None))
                return left
            arcs.append(Arc(int(right.label), int(left.label), None))
            return right
        
        def encode_bht(self, tree: PTB.Tree, tags: List[str], fences: List[str]) -> List[str]:
            if len(tree.deps) == 1:
                return ['>'], ['>'], [tree.label]
            left, right = tree.deps 
            if left.is_preterminal():
                left_tags, left_fences, left_cons = ['>'], [],  []
            else:
                left_tags, left_fences, left_cons = self.encode_bht(left, [], ['>'])
            if right.is_preterminal():
                right_tags, right_fences, right_cons = ['<'], [], []
            else:
                right_tags, right_fences, right_cons = self.encode_bht(right, [], ['<'])
            return left_tags + tags + right_tags, left_fences + fences + right_fences, left_cons + [tree.label] + right_cons

        def decode_bht(self, tags: List[str], fences: List[str], cons: List[str], n: int) -> PTB.Tree:
            pos = list(map(str, range(1, n+1)))
            stack = []
            for tag, fence, con in zip(tags, fences, cons):
                if tag == '>':
                    stack.append(PTB.Tree(self.ROOT, deps=[PTB.Tree(pos.pop(0))]))
                else:
                    stack[-1].deps.append(PTB.Tree(pos.pop(0)))
                    stack.pop(-1)
                if fence == '>':
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
            stack[-1].deps.append(PTB.Tree(pos.pop(0)))
            return stack[0].deps[0]
        
        def decode_postprocess_bht(
            self, 
            tags: List[str],
            fences: List[str],
            cons: List[str], 
            n: int
        ) -> Tuple[PTB.Tree, bool]:
            pos = list(map(str, range(1, n+1)))
            stack, well_formed = [], True
            for tag, fence, con in zip(tags, fences, cons):
                if tag == '>' or len(stack) <= 1:
                    stack.append(PTB.Tree(self.ROOT, deps=[PTB.Tree.from_leaf(pos[0], pos.pop(0))]))
                    well_formed &= (tag == '>')
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(pos[0], pos.pop(0)))
                    stack.pop(-1)
                if fence == '>' or len(stack) <= 2:
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                    well_formed &= (fence == '>')
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
            stack[-1].deps.append(PTB.Tree.from_leaf(pos[0], pos.pop(0)))
            if len(stack) == 1:
                tree = stack[0].deps[0]
            else:
                stack = [t.deps[0] for t in stack if t.label == self.ROOT]
                while len(stack) > 1:
                    last = stack.pop(-1)
                    stack[-1].deps.append(last)
                tree = stack[0]
                well_formed &= (len(stack) == 1)
            return tree, well_formed
        
        def test(self, graph: CoNLL.Graph) -> bool:
            planar = graph.planarize(1)
            labels = self.encode(planar)
            rec1 = planar.rebuild(self.decode(*labels, planar.DEPREL))
            rec2, well_formed = self.decode_postprocess(*labels, planar.DEPREL)
            return planar == rec1 == planar.rebuild(rec2) and well_formed
    
    def __init__(
        self,
        model: HexaTaggingDependencyModel,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        proj: Optional[str],
        device: Union[int, str]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.labeler = self.Labeler(proj)
        self.proj = proj 
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--proj', default=None, type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 
    
    @classmethod
    def transform(
        cls,
        graph: CoNLL.Graph,
        input_tkzs: List[Tokenizer],
        HEXA: Tokenizer,
        FENCE: Tokenizer,
        CON: Tokenizer,
        REL: Tokenizer,
        labeler: HexaTaggingDependencyParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            hexas, fences, cons = labeler.encode(graph)
            graph.HEXA = HEXA.encode(hexas).pin_memory()
            graph.FENCE = FENCE.encode(fences).pin_memory()
            graph.CON = CON.encode(cons).pin_memory()
            graph.REL = REL.encode(graph.DEPREL).pin_memory()
            graph._transformed = True 
            
    def collate(self, graphs: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.target_tkzs]
        lens = list(map(len, graphs))
        mask0 = create_mask(lens)
        mask1 = mask0.clone()
        for i, l in enumerate(lens):
            mask1[i, max(l-1,1):] = False
        return inputs, targets, [mask0, mask1], graphs
    
    def _pred(
        self, 
        graph: CoNLL.Graph, 
        hexa_pred: torch.Tensor,
        fence_pred: torch.Tensor,
        con_pred: torch.Tensor,
        rel_pred: torch.Tensor
    ) -> Tuple[CoNLL.Graph, bool]:
        rec, well_formed = self.labeler.decode_postprocess(
            self.HEXA.decode(hexa_pred),
            self.FENCE.decode(fence_pred),
            self.CON.decode(con_pred),
            self.REL.decode(rel_pred))
        pred = graph.rebuild(rec)
        if self.proj:
            pred = pred.deprojectivize(self.proj)
        return graph.rebuild(rec), well_formed
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        hexas, fences, cons, rels = targets
        s_hexa, s_fence, s_con, s_rel = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss([s_hexa, s_fence, s_con, s_rel], targets)
        return loss, dict(HEXA=acc(s_hexa, hexas), FENCE=acc(s_fence, fences),
                          CON=acc(s_con, cons), REL=acc(s_rel, rels))    
        
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor],
        masks: List[torch.Tensor],
        graphs: List[CoNLL.Graph]
    ) -> Iterator[CoNLL.Graph]:
        lens0, lens1 = masks[0].sum(-1).tolist(), masks[1].sum(-1).tolist()
        hexa_preds, fence_preds, con_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, graphs, hexa_preds.split(lens1), fence_preds.split(lens1),
                            con_preds.split(lens1), rel_preds.split(lens0)))
        return preds
    
    @torch.no_grad()
    def control_step(
        self,
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        lens0, lens1 = masks[0].sum(-1).tolist(), masks[1].sum(-1).tolist()
        hexas, fences, cons, rels = targets
        loss, hexa_preds, fence_preds, con_preds, rel_preds = self.model.control(inputs[0], inputs[1:], targets, *masks)
        preds, well_formed = zip(*map(self._pred, graphs, hexa_preds.split(lens1), fence_preds.split(lens1),
                            con_preds.split(lens1), rel_preds.split(lens0)))
        control = dict(
            well_formed=avg(well_formed)*100, loss=loss.item(),
            HEXA=acc(hexa_preds, hexas), FENCE=acc(fence_preds, fences),
            CON=acc(con_preds, cons), REL=acc(rel_preds, rels))
        return control, preds
    
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        proj: Optional[str] = None,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> HexaTaggingDependencyParser:
        if isinstance(data, str):
            data = cls.load_data(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'UPOS'))
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
                conf.update(**tkz.conf())
            
        hexa_tkz, fence_tkz, con_tkz, rel_tkz = Tokenizer('HEXA'), Tokenizer('FENCE'), Tokenizer('CON'), Tokenizer('REL')
        labeler = cls.Labeler(proj)
        if pretest:
            assert all(parallel(labeler.test, data, name=f'{cls.NAME}[pretest]'))
        hexas, fences, cons = zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]'))
        hexa_tkz.train(*flatten(hexas))
        fence_tkz.train(*flatten(fences))
        con_tkz.train(*flatten(cons))
        rel_tkz.train(*[arc.REL for graph in data for arc in graph.arcs])
        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        model = cls.MODEL(enc_conf, *in_confs, hexa_tkz.conf, fence_tkz.conf, con_tkz.conf, rel_conf).to(device)
        return cls(model, input_tkzs, [hexa_tkz, fence_tkz, con_tkz, rel_tkz], proj, device)
