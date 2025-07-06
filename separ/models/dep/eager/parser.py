from __future__ import annotations
from typing import Iterable 
from argparse import ArgumentParser
import torch

from separ.parser import Parser 
from separ.models.dep.eager.model import ArcEagerDependencyModel
from separ.utils import DependencyMetric, get_mask, acc, Config, bar, flatten, ControlMetric
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.data.struct import Arc, adjacent_from_arcs, candidates_no_cycles

class ArcEagerDependencyParser(Parser):
    """Transition-based Arc-Eager Dependency Parser from [Nivre and Fernández-González, 2012](https://aclanthology.org/J14-2002.pdf)."""
    NAME = 'dep-eager'
    MODEL = ArcEagerDependencyModel
    PARAMS = ['stack', 'buffer', 'proj']
    DATASET = [CoNLL]
    
    class State:
        def __init__(self, stack: list[int], buffer: list[int]):
            """Represents a state in the arc-eager transition-based system.

            Args:
                stack (list[int]): Top nodes in the stack.
                buffer (list[int]): Front nodes in the buffer.
            """
            self.stack = stack 
            self.buffer = buffer 
            
        @property
        def tensor(self) -> torch.Tensor:
            return torch.tensor(self.stack + self.buffer)
    
    class Oracle:
        ACTIONS = ['left-arc', 'right-arc', 'shift', 'reduce']
        
        def __init__(self, stack: int, buffer: int):
            self.n_stack = stack 
            self.n_buffer = buffer 
        
        def reset(self, n: int) -> ArcEagerDependencyParser.Oracle:
            self.stack: list[int] = [0]
            self.buffer: list[int] = list(range(1, n+1))
            self.n = n 
            self.assigned: set[int] = set()
            self.recovered: set[Arc] = set()
            self.actions: list[str] = []
            self.rels: list[str] = []
            self.root = False 
            return self 
            
        def left_arc_valid(self) -> bool:
            return self.stack[-1] != 0 and self.stack[-1] not in self.assigned
        
        def right_arc_valid(self) -> bool:
            return self.buffer[0] not in self.assigned and not (self.stack[-1] == 0 and self.root)
        
        def reduce_valid(self) -> bool:
            return self.stack[-1] in self.assigned 
        
        def left_arc(self, REL: str) -> ArcEagerDependencyParser.State:
            arc = Arc(self.buffer[0], self.stack[-1], REL)
            self.recovered.add(arc)
            self.assigned.add(self.stack.pop(-1))
            return self.state
            
        def right_arc(self, REL: str) -> ArcEagerDependencyParser.State:
            arc = Arc(self.stack[-1], self.buffer[0], REL)
            self.root = self.root or (arc.HEAD == 0)
            self.recovered.add(arc)
            self.assigned.add(self.buffer[0])
            self.stack.append(self.buffer.pop(0))
            return self.state 
        
        def reduce(self) -> ArcEagerDependencyParser.State:
            self.stack.pop(-1)
            return self.state 
        
        def shift(self) -> ArcEagerDependencyParser.State:
            self.stack.append(self.buffer.pop(0))
            return self.state 
        
        @property 
        def state(self) -> ArcEagerDependencyParser.State:
            stack = self.stack[-min(len(self.stack), self.n_stack):]
            if len(stack) < self.n_stack:
                stack = [0 for _ in range(self.stack - len(stack))] + stack
            buffer = self.buffer[:min(self.n_buffer, len(self.buffer))]
            if len(buffer) < self.n_buffer:
                buffer += [self.n+1 for _ in range(self.n_buffer - len(buffer))]
            return ArcEagerDependencyParser.State(stack, buffer)
        
        @property
        def final(self) -> bool:
            return len(self.buffer) == 0
        
        def __call__(self, actions: list[str], REL: str) -> ArcEagerDependencyParser.State:
            done = False 
            for action in actions:
                if action == 'left-arc' and self.left_arc_valid():
                    done = True 
                    return self.left_arc(REL)
                elif action == 'right-arc' and self.right_arc_valid():
                    done = True 
                    return self.right_arc(REL)
                elif action == 'reduce' and self.reduce_valid():
                    done = True 
                    return self.reduce()
                elif action == 'shift':
                    done = True 
                    return self.shift()
            if not done:
                raise AssertionError('No action executed')
            
    class System:
        EMPTY_REL = '<pad>'
        DEFAULT_REL = 'punct'
            
        def __init__(self, stack: int, buffer: int, proj: str | None):
            self.stack = stack 
            self.buffer = buffer 
            self.proj = proj 
            
        def __repr__(self):
            return f'ArcEagerSystem(stack={self.stack}, buffer={self.buffer}, proj={self.proj})'
            
        def encode(self, tree: CoNLL.Tree) -> tuple[list[ArcEagerDependencyParser.State], list[str], list[str]]:
            if self.proj:
                tree = tree.projectivize(self.proj)
            oracle = ArcEagerDependencyParser.Oracle(self.stack, self.buffer)
            arcs: dict[int, Arc] = {arc.DEP: arc for arc in tree.arcs}
            states, actions, rels = [], [], []
            oracle.reset(len(tree))
            while not oracle.final:
                states.append(oracle.state)
                if oracle.left_arc_valid() and arcs[oracle.stack[-1]].HEAD == oracle.buffer[0]:
                    action, rel = 'left-arc', arcs[oracle.stack[-1]].REL
                    oracle.left_arc(arcs[oracle.stack[-1]].REL)
                elif oracle.right_arc_valid() and arcs[oracle.buffer[0]].HEAD == oracle.stack[-1]:
                    action, rel = 'right-arc', arcs[oracle.buffer[0]].REL
                    oracle.right_arc(arcs[oracle.buffer[0]].REL)
                elif oracle.reduce_valid() and oracle.stack[-1] not in [arcs[b].HEAD for b in oracle.buffer]:
                    action, rel = 'reduce', self.EMPTY_REL
                    oracle.reduce()
                else:
                    action, rel = 'shift', self.EMPTY_REL
                    oracle.shift()
                actions.append(action)
                rels.append(rel)
            return states, actions, rels 
                    
        def decode(self, n: int, actions: list[str], rels: list[str]) -> list[Arc]:
            oracle = ArcEagerDependencyParser.Oracle(self.stack, self.buffer)
            oracle.reset(n)
            for action, rel in zip(actions, rels):
                if action == 'left-arc':
                    oracle.left_arc(rel)
                elif action == 'right-arc':
                    oracle.right_arc(rel)
                elif action == 'shift':
                    oracle.shift()
                elif action == 'reduce':
                    oracle.reduce()
                else:
                    raise NotImplementedError
            return sorted(oracle.recovered)
        
        def test(self, tree: CoNLL.Tree) -> bool:
            if len(tree.planes) > 1:
                tree = tree.projectivize()
            _, actions, rels = self.encode(tree)
            recovered = tree.rebuild_from_arcs(self.decode(len(tree), actions, rels))
            return recovered == tree
        
        def postprocess(self, arcs: list[Arc], n: int) -> list[Arc]:
            no_assigned = set(range(1, n+1)) - set(arc.DEP for arc in arcs)
            adjacent = adjacent_from_arcs(arcs, n)
            for dep in no_assigned:
                head = candidates_no_cycles(adjacent, n).pop(0)
                adjacent[dep, head] = True 
                arcs.append(Arc(head, dep, self.DEFAULT_REL))
            return arcs 
        
                
    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        stack: int,
        buffer: int,
        proj: str | None,
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.stack, self.buffer, self.proj = stack, buffer, proj 
        self.system = self.System(stack, buffer, proj)
        
    @property 
    def METRIC(self) -> DependencyMetric:
        return DependencyMetric()
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--stack', type=int, default=1, help='Number of tokens from the stack used to represent a state')
        argparser.add_argument('--buffer', type=int, default=1, help='Number of tokens from the buffer used to represent a state')
        argparser.add_argument('--proj', default=None, type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            states, tree.ACTION, tree.REL = self.system.encode(tree)
            tree.STATE = torch.stack([state.tensor for state in states])
            tree.tranformed = True 
        return tree
            
    def collate(self, trees: list[CoNLL.Tree]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[CoNLL.Tree]]:
        inputs = [tkz.batch_encode(trees, pin=False) for tkz in self.input_tkzs]
        states = [tree.STATE for tree in trees]
        targets = [tkz.batch_encode(trees, mode='cat', pin=False) for tkz in self.target_tkzs]
        masks = [
            get_mask(list(map(len, trees))), get_mask([state.shape[0] for state in states])
        ]
        return inputs, masks, [states, *targets], trees
            
    def _pred(self, tree: CoNLL.Tree, oracle: Oracle) -> CoNLL.Tree:
        arcs = self.system.postprocess(list(oracle.recovered), len(tree))
        pred = tree.rebuild_from_arcs(arcs)
        if self.proj:
            pred = pred.deprojectivize(self.proj)
        return pred 
            
    def train_step(
        self,
        model: ArcEagerDependencyModel,
        inputs: list[torch.Tensor], 
        _: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        states, actions, rels = targets
        s_action, s_rel = model(inputs[0], inputs[1:], states)
        loss = model.loss(s_action, s_rel, actions, rels)
        return loss, ControlMetric(loss=loss.detach(), ACTION=acc(s_action, actions), REL=acc(s_rel, rels))
        
    @torch.no_grad()
    def pred_step(
        self, 
        model: ArcEagerDependencyModel,
        inputs: list[torch.Tensor], 
        _: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> Iterable[CoNLL.Tree]:
        embed = model.encode(inputs[0], *inputs[1:])
        oracles = [self.Oracle(self.stack, self.buffer, self.proj).reset(len(tree)) for tree in trees]
        while not all(oracle.final for oracle in oracles):
            states = [oracle.state.tensor.to(self.device).unsqueeze(0) for oracle in oracles]
            action_preds, rel_preds = model.predict(embed, states)
            action_preds = list(map(self.ACTION.decode, action_preds.unbind(0)))
            rel_preds = self.REL.decode(rel_preds.flatten())
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        return map(self._pred, trees, oracles)
    
    @torch.no_grad()
    def eval_step(
        self, 
        model: ArcEagerDependencyModel,
        inputs: list[torch.Tensor], 
        _: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, list[CoNLL.Tree]]:
        states, actions, rels = targets
        loss, embed, action_preds, rel_preds = model.control(inputs[0], inputs[1:], states)
        control = ControlMetric(loss=loss.detach(), ACTION=acc(action_preds, actions), REL=acc(rel_preds, rels))
        oracles = [self.Oracle(self.stack, self.buffer).reset(len(tree)) for tree in trees]
        while not all(oracle.final for oracle in oracles):
            states = [oracle.state.tensor.to(self.device).unsqueeze(0) for oracle in oracles]
            action_preds, rel_preds = model.predict(embed, states)
            action_preds = list(map(self.ACTION.decode, action_preds.unbind(0)))
            rel_preds = self.REL.decode(rel_preds.flatten())
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        return control, map(self._pred, trees, oracles)


    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        stack: int = 1,
        buffer: int = 1,
        proj: str | None = None,
        device: int = 0,
        **_
    ) -> ArcEagerDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=True, eos=True)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM', bos=True, eos=True)]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS', bos=True, eos=True))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM', bos=True, eos=True))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)

            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
        
        # train target tokenizers 
        action_tkz, rel_tkz = TargetTokenizer('ACTION'), TargetTokenizer('REL')
        system = cls.System(stack, buffer, proj)
        _, actions, rels = map(flatten, zip(*bar(map(system.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        action_tkz.train(actions)
        rel_tkz.train(rels)
        return cls(input_tkzs, [action_tkz, rel_tkz], [enc_conf, *in_confs, action_tkz.conf, rel_tkz.conf], stack, buffer, proj, device)
        
            
        
    