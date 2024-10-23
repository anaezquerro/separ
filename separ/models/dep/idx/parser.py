from __future__ import annotations 
from typing import List, Tuple, Union, Optional, Dict, Iterator
from argparse import ArgumentParser
import torch 


from separ.models.dep.idx.model import IndexDependencyModel
from separ.utils import Config, flatten, parallel, DependencyMetric, create_mask, acc, avg
from separ.data import CoNLL, AbstractTokenizer, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.structs import Arc, forms_cycles, adjacent_from_arcs
from separ.parser import Parser
from separ.models.dep.labeler import DependencyLabeler


class IndexDependencyParser(Parser):
    """Index Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)"""
    NAME = 'dep-idx'
    DATASET = CoNLL 
    METRIC = DependencyMetric
    MODEL = IndexDependencyModel
    PARAMS = ['rel']
    
    class Labeler(DependencyLabeler):
        DEFAULT = '-1'
        
        def __init__(self, rel: bool = False):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexDependencyLabeler(rel={self.rel})'
            
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
            indexes, rels = ['' for _ in range(len(graph))], ['' for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1] = str(arc.HEAD - (arc.DEP*self.rel))
                rels[arc.DEP-1] = arc.REL 
            return indexes, rels 
        
        def decode(self, indexes: List[str], rels: List[str]) -> List[Arc]:
            arcs, root = [], False
            for idep, index in enumerate(indexes):
                dep = idep + 1
                head = int(index)+dep*self.rel 
                arcs.append(Arc(head, dep, rels[dep-1]))
                root = root or (head == 0)
            return arcs 
        
        def decode_postprocess(self, indexes: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
            n, arcs, root = len(indexes), [], False
            adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
            well_formed = True
            for idep, index in enumerate(indexes):
                dep = idep + 1
                head = int(index)+dep*self.rel 
                if head not in range(0, n+1) or head == dep or forms_cycles(adjacent, dep, head) or (root and head == 0):
                    well_formed = False 
                    candidates = set(range(1*root, n+1)) - {head, dep}
                    head = [h for h in candidates if not forms_cycles(adjacent, dep, h)].pop(0)
                arcs.append(Arc(head, dep, rels[dep-1]))
                adjacent[dep, head] = True 
                root = root or (head == 0)
            return arcs, well_formed
        
        def test(self, graph: CoNLL.Graph) -> bool:
            indexes, rels = self.encode(graph)
            rec1 = self.decode(indexes, rels)
            rec2, well_formed = self.decode_postprocess(indexes, rels)
            rec1 = adjacent_from_arcs(rec1, len(graph))
            rec2 = adjacent_from_arcs(rec2, len(graph))
            return bool(((graph.ADJACENT == rec1) & (graph.ADJACENT == rec2)).all()) and well_formed
        
    def __init__(
        self,
        model: IndexDependencyModel, 
        input_tkzs: List[AbstractTokenizer],
        target_tkzs: List[AbstractTokenizer],
        rel: bool,
        device: Union[str, int]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.rel = rel
        self.labeler = self.Labeler(rel)
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
            
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    
    @classmethod
    def transform(
        cls, 
        graph: CoNLL.Graph, 
        input_tkzs: List[Tokenizer], 
        INDEX: Tokenizer, 
        REL: Tokenizer, 
        labeler: IndexDependencyParser.Labeler
    ):
        """Transformation step to collate inputs.

        Args:
            graph (CoNLL.Graph): Input dependency graph.
            input_tkzs (List[Tokenizer]): Input tokenizers.
            INDEX (Tokenizer): Tokenizer of the index component.
            REL (Tokenizer): Tokenizer of the arc relation (2nd component).
            labeler (SLDependencyParser.Labeler): Dependency labeler.
        """
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            indexes, rels = labeler.encode(graph)
            graph.INDEX = INDEX.encode(indexes).pin_memory()
            graph.REL = REL.encode(rels).pin_memory()
            graph._transformed = True 
            
    def collate(self, graphs: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.target_tkzs]
        masks = [create_mask(list(map(len, graphs)))]
        return inputs, targets, masks, graphs
    
    def _pred(
        self,
        graph: CoNLL.Graph, 
        index_pred: torch.Tensor, 
        rel_pred: torch.Tensor
    ) -> Tuple[CoNLL.Graph, bool]:
        """Performs dependency graph reconstruction.

        Args:
            graph (CoNLL.Graph): Input dependency graph.
            index_pred (torch.Tensor ~ seq_len): Index prediction.
            rel_pred (torch.Tensor ~ seq_len): Relation prediction.

        Returns:
            Tuple[CoNLL.Graph, bool]: Predicted dependency graph and whether the sequence of 
                components conforms a well-formed dependency graph.
        """
        rec, well_formed = self.labeler.decode_postprocess(self.INDEX.decode(index_pred), self.REL.decode(rel_pred))
        return graph.rebuild(rec), well_formed
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Loss and control metrics.
        """
        indexes, rels = targets
        s_index, s_rel = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_index, s_rel, indexes, rels)
        return loss, dict(INDEX=acc(s_index, indexes), REL=acc(s_rel, rels))
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Iterator[CoNLL.Graph]:
        """Prediction step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[CoNLL.Graph]): List of input dependency graphs.

        Returns:
            Iterator[CoNLL.Graph]: Predicted dependency graph.
        """
        lens = masks[0].sum(-1).tolist()
        index_preds, rel_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, graphs, index_preds.split(lens), rel_preds.split(lens)))
        return preds  
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        """Control (prediction + evaluation) step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[CoNLL.Graph]): List of input dependency graphs.

        Returns:
            Tuple[Dict[str, float], Iterator[CoNLL.Graph]]: Control metrics and predicted dependency graphs.
        """
        indexes, rels, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, index_preds, rel_preds = self.model.control(inputs[0], inputs[1:], *targets, mask)
        preds, well_formed = zip(*map(self._pred, graphs, index_preds.split(lens), rel_preds.split(lens)))
        control = dict(INDEX=acc(index_preds, indexes), REL=acc(rel_preds, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
        return control, preds
        
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        rel: bool = False,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> IndexDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data, num_workers)
        
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
            
        # train target tokenizers 
        index_tkz, rel_tkz = Tokenizer('INDEX'), Tokenizer('REL')
        labeler = cls.Labeler(rel=rel)
        if pretest:
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        indexes, rels = map(flatten, zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        index_tkz.train(*indexes)
        rel_tkz.train(*rels)
            
        model = cls.MODEL(enc_conf, *in_confs, index_tkz.conf, rel_tkz.conf).to(device)
        return cls(model, input_tkzs, [index_tkz, rel_tkz], rel, device)
        
            