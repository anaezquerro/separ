from __future__ import annotations
import torch 
from typing import Set, List, Tuple, Iterable, Union, Optional

from separ.parser import Parser 
from separ.models.sdp.cov.model import CovingtonSemanticModel
from separ.data import SDP, EnhancedCoNLL, Arc, Graph, Dataset, InputTokenizer, TargetTokenizer, CharacterTokenizer, PretrainedTokenizer, adjacent_from_arcs
from separ.utils import Config, SemanticMetric, get_mask, ControlMetric, acc, flatten, bar, pad2D, to 

class CovingtonSemanticParser(Parser):
    """Transition-based Covington Semantic Parser from [Covington (2001)](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf)."""
    NAME = 'sdp-cov'
    MODEL = CovingtonSemanticModel
    PARAMS = []
    DATASET = [SDP, EnhancedCoNLL]
    
    class State:
        def __init__(self, p1: int, p2: int):
            """Represents a state of the Covington system.
            
            Args:
                p1 (int): First pointer.
                p2 (int): Second pointer.
                recovered (Set[Arc]): Recovered arcs in the current state.
            """
            self.p1 = p1 
            self.p2 = p2 
            self.pointer = torch.tensor([self.p1, self.p2])

    class Oracle:
        ACTIONS = ['left-arc', 'right-arc', 'no-arc', 'shift']
        
        def __init__(self):
            ...
            
        def reset(self, n: int) -> CovingtonSemanticParser.Oracle:
            self.recovered: Set[Arc] = set()
            self.matrix = torch.zeros(n+2, n+2, dtype=bool)
            self.p1 = 0
            self.p2 = 1 
            self.n = n
            return self 
        
        def left_arc(self, REL: str) -> CovingtonSemanticParser.State:
            arc = Arc(self.p2, self.p1, REL)
            self.recovered.add(arc)
            self.matrix[arc.DEP, arc.HEAD] = True
            return self.no_arc()
        
        def left_arc_valid(self) -> bool:
            return not self.matrix[self.p1, self.p2]

        def right_arc(self, REL: str) -> CovingtonSemanticParser.State:
            arc = Arc(self.p1, self.p2, REL)
            self.recovered.add(arc)
            self.matrix[arc.DEP, arc.HEAD] = True
            return self.no_arc()
        
        def right_arc_valid(self) -> bool:
            return not self.matrix[self.p2, self.p1]
        
        def no_arc(self) -> CovingtonSemanticParser.State:
            if self.p1 > 0:
                self.p1 -= 1
            else:
                self.shift()
            return self.state
                
        def no_arc_valid(self) -> bool:
            return self.p1 >= 0
                
        def shift(self) -> CovingtonSemanticParser.State:
            self.p2 += 1 
            self.p1 = self.p2-1
            return self.state 
        
        @property
        def state(self) -> CovingtonSemanticParser.State:
            return CovingtonSemanticParser.State(self.p1, self.p2)
        
        def __call__(self, actions: List[str], REL: str) -> CovingtonSemanticParser.State:
            for action in actions:
                if self.left_arc_valid() and action == 'left-arc':
                    return self.left_arc(REL)
                elif self.right_arc_valid() and action == 'right-arc':
                    return self.right_arc(REL)
                elif self.no_arc_valid() and action == 'no-arc':
                    return self.no_arc()
                else:
                    return self.shift() 
            raise AssertionError
            
        @property
        def final(self):
            return self.p2 > self.n
                
    class System:
        EMPTY_REL = '<pad>'
        DEFAULT_REL = 'punct'
        
        def __init__(self):
            ... 
            
        def __repr__(self):
            return f'CovingtonSystem()'
        
        def encode(self, graph: Graph) -> Tuple[List[CovingtonSemanticParser.State], List[str], List[str]]:
            oracle = CovingtonSemanticParser.Oracle()
            oracle.reset(len(graph))
            states, actions, rels = [], [], []
            while not oracle.final:
                states.append(oracle.state)
                if oracle.left_arc_valid() and graph.ADJACENT[oracle.p1, oracle.p2]:
                    action, rel = 'left-arc', graph.LABELED_ADJACENT[oracle.p1, oracle.p2]
                    oracle.left_arc(rel)
                elif oracle.right_arc_valid() and graph.ADJACENT[oracle.p2, oracle.p1]:
                    action, rel = 'right-arc', graph.LABELED_ADJACENT[oracle.p2, oracle.p1]
                    oracle.right_arc(rel)
                elif oracle.no_arc_valid() and (
                    graph.ADJACENT[:oracle.p1, oracle.p2].any() or graph.ADJACENT[oracle.p2, :oracle.p1].any()
                ):
                    action, rel = 'no-arc', self.EMPTY_REL
                    oracle.no_arc()
                else:
                    action, rel = 'shift', self.EMPTY_REL
                    oracle.shift()
                actions.append(action)
                rels.append(rel)
            return states, actions, rels
        
        def decode(self, n: int, actions: List[str], rels: List[str]) -> List[Arc]:
            oracle = CovingtonSemanticParser.Oracle()
            oracle.reset(n)
            for action, rel in zip(actions, rels):
                if action == 'left-arc':
                    oracle.left_arc(rel)
                elif action == 'right-arc':
                    oracle.right_arc(rel)
                elif action == 'no-arc':
                    oracle.no_arc()
                else:
                    oracle.shift()
            return sorted(oracle.recovered)

        def test(self, graph: Graph) -> bool:
            _, actions, rels = self.encode(graph)
            recovered = graph.rebuild_from_arcs(self.decode(len(graph), actions, rels))
            return recovered == graph 
        
        
    def __init__(
        self,
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.system = self.System()
        
    @property 
    def METRIC(self) -> SemanticMetric:
        return SemanticMetric()
    
    def loader(self, data: Dataset, batch_size: int, shuffle: bool):
        _, sampler = super().loader(data, batch_size, shuffle)
        lens = {i: max(len(graph), graph.POINTER.shape[0]) for i, graph in enumerate(data)}
        return data.loader(
            batch_size=batch_size, 
            shuffle=shuffle,
            collate=self.collate, 
            device=sampler.device, 
            lens=lens
        )
    
    def transform(self, graph: Graph) -> Graph:
        if not graph.transformed:
            states, graph.ACTION, graph.REL = self.system.encode(graph)
            graph.POINTER = torch.stack([state.pointer for state in states])
            graph.MATRIX = torch.zeros(len(graph)+2, len(graph)+2, dtype=torch.bool)
            graph.MATRIX[:-1, :-1] = graph.ADJACENT
            graph.transformed = True 
        return graph
            
    def collate(self, graphs: List[Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Graph]]:
        inputs = [tkz.batch_encode(graphs, pin=False) for tkz in self.input_tkzs]
        pointers = torch.cat([graph.POINTER for graph in graphs])
        pointers = torch.cat([
            torch.tensor(flatten([i]*len(graph.POINTER) for i, graph in enumerate(graphs))).unsqueeze(-1),
            pointers
        ], dim=1)
        matrices = pad2D(graph.MATRIX for graph in graphs)
        targets = [tkz.batch_encode(graphs, mode='cat', pin=False) for tkz in self.target_tkzs]
        return inputs, [targets[-1] != self.REL.pad_index], [pointers, matrices, *targets], graphs
            
    def _pred(self, graph: Graph, oracle: CovingtonSemanticParser.Oracle):
        return graph.rebuild_from_arcs(oracle.recovered)
            
    def train_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ControlMetric]:
        pointers, matrices, actions, rels, mask = *targets, *masks
        s_action, s_rel = self.model(inputs[0], inputs[1:], pointers, matrices)
        s_rel, rels = s_rel[mask], rels[mask]
        loss = self.model.loss(s_action, s_rel, actions, rels)
        return loss, ControlMetric(
            loss=loss.detach(), 
            ACTION=acc(s_action, actions), REL=acc(s_rel, rels)
        )
        
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        _: List[torch.Tensor], 
        graphs: List[Graph]
    ) -> Iterable[Graph]:
        embed = self.model.encode(inputs[0], *inputs[1:])
        return self._pred_step(embed, graphs)

    @torch.no_grad()
    def _pred_step(
        self,
        embed: torch.Tensor,
        graphs: List[Graph]
    ) -> Iterable[Graph]:
        oracles = [self.Oracle().reset(len(graph)) for graph in graphs]
        while not all(oracle.final for oracle in oracles):
            pointers = torch.tensor([[i, *oracle.state.pointer] for i, oracle in enumerate(oracles)])
            matrices = pad2D(oracle.matrix for oracle in oracles)
            action_preds, rel_preds = self.model.predict(embed, *to(embed.device, pointers, matrices))
            action_preds = list(map(self.ACTION.decode, action_preds.unbind(0)))
            rel_preds = self.REL.decode(rel_preds.flatten())
            for oracle, action_pred, rel_pred in zip(oracles, action_preds, rel_preds):
                if not oracle.final:
                    oracle(action_pred, rel_pred)
        return map(self._pred, graphs, oracles)
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        graphs: List[Graph]
    ) -> Tuple[ControlMetric, SemanticMetric]:
        *_, actions, rels, mask = *targets, *masks
        loss, embed, s_action, s_rel = self.model.control(inputs[0], inputs[1:], mask, *targets)
        control = ControlMetric(
            loss=loss.detach(), 
            ACTION=acc(s_action, actions), 
            REL=acc(s_rel[mask], rels[mask])
        )
        return control, self.METRIC(self._pred_step(embed, graphs), graphs)
    
    @classmethod 
    def build(
        cls, 
        data: Union[SDP, EnhancedCoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        device: int = 0,
        **_
    ) -> CovingtonSemanticParser:
        if isinstance(data, str):
            data = cls.load_data(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=True, eos=True)]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM', bos=True, eos=True)]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'POS', bos=True, eos=True))
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
        system = cls.System()
        _, actions, rels = map(flatten, zip(*bar(map(system.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        action_tkz.train(actions)
        rel_tkz.train(rels)
        return cls(input_tkzs, [action_tkz, rel_tkz], [enc_conf, *in_confs, action_tkz.conf, rel_tkz.conf], device)
        
            
        
    