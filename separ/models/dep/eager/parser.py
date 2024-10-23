from __future__ import annotations
from typing import List, Tuple, Dict, Union, Optional, Set
from argparse import ArgumentParser
import torch, os 

from separ.parser import Parser
from separ.models.dep.eager.model import ArcEagerDependencyModel
from separ.utils import DependencyMetric, create_mask, acc, Config, parallel, flatten, split
from separ.data import CoNLL, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.structs import Arc, adjacent_from_arcs, candidates_no_cycles

class ArcEagerDependencyParser(Parser):
    NAME = 'dep-eager'
    MODEL = ArcEagerDependencyModel
    METRIC = DependencyMetric
    DATASET = CoNLL 
    PARAMS = ['n_stack', 'n_buffer']
    
    class State:
        def __init__(self, stack: List[int], buffer: List[int]):
            """Represents a state in the arc-eager transition-based system.

            Args:
                stack (List[int]): Top nodes in the stack.
                buffer (List[int]): Front nodes in the buffer.
            """
            self.stack = stack 
            self.buffer = buffer 
            
        @property
        def tensor(self) -> torch.Tensor:
            return torch.tensor(self.stack + self.buffer)
    
    class Oracle:
        ACTIONS = ['left-arc', 'right-arc', 'shift', 'reduce']
        
        def __init__(self, n_stack: int, n_buffer: int):
            self.n_stack = n_stack 
            self.n_buffer = n_buffer 
        
        def reset(self, n: int) -> ArcEagerDependencyParser.Oracle:
            self.stack: List[int] = [0]
            self.buffer: List[int] = list(range(1, n+1))
            self.n = n 
            self.assigned: Set[int] = set()
            self.recovered: Set[Arc] = set()
            self.actions: List[str] = []
            self.rels: List[str] = []
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
                stack = [0 for _ in range(self.n_stack - len(stack))] + stack
            buffer = self.buffer[:min(self.n_buffer, len(self.buffer))]
            if len(buffer) < self.n_buffer:
                buffer += [self.n+1 for _ in range(self.n_buffer - len(buffer))]
            return ArcEagerDependencyParser.State(stack, buffer)
        
        @property
        def final(self) -> bool:
            return len(self.buffer) == 0
        
        def __call__(self, actions: List[str], REL: str) -> ArcEagerDependencyParser.State:
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
            
        def __init__(self, n_stack: int, n_buffer: int):
            self.n_stack = n_stack 
            self.n_buffer = n_buffer 
            
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[ArcEagerDependencyParser.State], List[str], List[str]]:
            oracle = ArcEagerDependencyParser.Oracle(self.n_stack, self.n_buffer)
            arcs: Dict[int, Arc] = {arc.DEP: arc for arc in graph.arcs}
            states, actions, rels = [], [], []
            oracle.reset(len(graph))
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
                
                    
        def decode(self, n: int, actions: List[str], rels: List[str]) -> List[Arc]:
            oracle = ArcEagerDependencyParser.Oracle(self.n_stack, self.n_buffer)
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
        
        def test(self, graph: CoNLL.Graph) -> bool:
            if len(graph.planes) > 1:
                return True 
            _, actions, rels = self.encode(graph)
            recovered = graph.rebuild(self.decode(len(graph), actions, rels))
            return recovered == graph
        
        def postprocess(self, arcs: List[Arc], n: int) -> List[Arc]:
            no_assigned = set(range(1, n+1)) - set(arc.DEP for arc in arcs)
            adjacent = adjacent_from_arcs(arcs, n)
            for dep in no_assigned:
                head = candidates_no_cycles(adjacent, n).pop(0)
                adjacent[dep, head] = True 
                arcs.append(Arc(head, dep, self.DEFAULT_REL))
            return arcs 
        
                
    def __init__(
            self,
            model: ArcEagerDependencyModel, 
            input_tkzs: List[Tokenizer],
            target_tkzs: List[Tokenizer],
            n_stack: int, 
            n_buffer: int,
            device: str
        ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.n_stack, self.n_buffer = n_stack, n_buffer
        self.system = self.System(n_stack, n_buffer)
        self.TRANSFORM_ARGS = [input_tkzs, target_tkzs, self.system]
        
    def __repr__(self) -> str:
        return f'ArcEagerDependencyParser(n_stack={self.n_stack}, n_buffer={self.n_buffer})'
    
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--n_stack', type=int, default=1, help='Number of tokens from the stack used to represent a state')
        argparser.add_argument('--n_buffer', type=int, default=1, help='Number of tokens from the buffer used to represent a state')
        return argparser 
    
    @classmethod
    def transform(cls, graph: CoNLL.Graph, input_tkzs: List[Tokenizer], ACTION: Tokenizer, REL: Tokenizer, system: ArcEagerDependencyParser.System):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            states, actions, rels = system.encode(graph)
            graph.ACTION = ACTION.encode(actions).pin_memory()
            graph.REL = REL.encode(rels).pin_memory()
            graph.STATE = torch.stack([state.tensor for state in states])
            graph._transformed = True 
            
    def collate(self, batch: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        assert all(graph._transformed for graph in batch), 'Dataset is not transformed'
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        states, actions, rels = zip(*[(graph.STATE, graph.ACTION, graph.REL) for graph in batch])
        masks = [create_mask(list(map(len, batch))), create_mask([state.shape[0] for state in states])]
        return inputs, [torch.cat(actions), torch.cat(rels), states], masks, batch
    
    def train_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor], _: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        actions, rels, states = targets
        s_action, s_rel = self.model(inputs[0], inputs[1:], states)
        loss = self.model.loss(s_action, s_rel, actions, rels)
        return loss, dict(aacc=acc(s_action, actions), racc=acc(s_rel, rels))
        
    @torch.no_grad()
    def pred_step(self, inputs: List[torch.Tensor], _: List[torch.Tensor], graphs: List[CoNLL.Graph]) -> List[CoNLL.Graph]:
        embed = self.model.encode(inputs[0], *inputs[1:])
        oracles = [self.Oracle(self.n_stack, self.n_buffer).reset(len(graph)) for graph in graphs]
        while not all(oracle.final for oracle in oracles):
            states = [oracle.state.tensor.to(self.device).unsqueeze(0) for oracle in oracles]
            action_preds, rel_preds = self.model.predict(embed, states)
            action_preds = list(map(self.ACTION.decode, action_preds.unbind(0)))
            rel_preds = self.REL.decode(rel_preds.flatten())
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        return [graph.rebuild(self.system.postprocess(list(oracle.recovered), len(graph))) for graph, oracle in zip(graphs, oracles)]
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        _: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], List[CoNLL.Graph]]:
        actions, rels, states = targets
        embed, s_action, s_rel = self.model.control(inputs[0], inputs[1:], states)
        loss = self.model.loss(s_action, s_rel, actions, rels)
        control = dict(loss=loss, aacc=acc(s_action, actions), racc=acc(s_rel, rels))
        oracles = [self.Oracle(self.n_stack, self.n_buffer).reset(len(graph)) for graph in graphs]
        while not all(oracle.final for oracle in oracles):
            states = [oracle.state.tensor.to(self.device).unsqueeze(0) for oracle in oracles]
            action_preds, rel_preds = self.model.predict(embed, states)
            action_preds = list(map(self.ACTION.decode, action_preds.unbind(0)))
            rel_preds = self.REL.decode(rel_preds.flatten())
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        preds = [graph.rebuild(self.system.postprocess(list(oracle.recovered), len(graph))) for graph, oracle in zip(graphs, oracles)]
        return control, preds
    
    @torch.no_grad()
    def ref_step(self, targets: Tuple[torch.Tensor], masks: Tuple[torch.Tensor], graphs: List[CoNLL.Graph]) -> List[CoNLL.Graph]:
        actions, rels, _, _, tmask = *targets, *masks 
        lens = tmask.sum(-1).tolist()
        oracles = [self.Oracle(self.n_stack, self.n_buffer).reset(len(graph)) for graph in graphs]
        actions, rels = split(self.ACTION.decode(actions), lens), split(self.REL.decode(rels), lens)
        while not all(oracle.final for oracle in oracles):
            action_preds = [[action.pop(0)] if len(action) > 0 else None for action in actions]
            rel_preds = [rel.pop(0) if len(rel) > 0 else None for rel in rels]
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        return [graph.rebuild(self.system.postprocess(list(oracle.recovered), len(graph))) for graph, oracle in zip(graphs, oracles)]


    @classmethod 
    def build(
            cls, 
            data: Union[CoNLL, str],
            enc_conf: Config,
            word_conf: Config, 
            tag_conf: Optional[Config] = None,
            char_conf: Optional[Config] = None,
            n_stack: int = 1,
            n_buffer: int = 1,
            pretest: bool = False,
            device: str = 'cuda:0',
            num_workers: int = os.cpu_count(),
            **_
        ) -> ArcEagerDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained, bos=True, eos=True)]
            word_conf |= input_tkzs[0].conf
            in_confs = [word_conf, None, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM', bos_token='<bos>', eos_token='<eos>')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(Tokenizer('TAG', 'UPOS', bos_token='<bos>', eos_token='<eos>'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM', bos_token='<bos>', eos_token='<eos>'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
            
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.update(vocab_size=len(tkz), pad_index=tkz.pad_index)
        
        # train target tokenizers 
        action_tkz, rel_tkz = Tokenizer('ACTION'), Tokenizer('REL')
        system = cls.System(n_stack, n_buffer)
        if pretest:
            assert all(parallel(system.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        _, actions, rels = map(flatten, zip(*parallel(system.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        action_tkz.train(*actions)
        rel_tkz.train(*rels)
        
        action_conf = Config(vocab_size=len(action_tkz), pad_index=action_tkz.pad_index)
        rel_conf = Config(vocab_size=len(rel_tkz), pad_index=rel_tkz.pad_index)
        model = cls.MODEL(enc_conf, *in_confs, action_conf, rel_conf, n_stack, n_buffer).to(device)
        model.confs = [enc_conf, *in_confs, action_conf, rel_conf, n_stack, n_buffer]
        return cls(model, input_tkzs, [action_tkz, rel_tkz], n_stack, n_buffer, device)
        
            
        
    