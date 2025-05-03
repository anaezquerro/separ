from __future__ import annotations
from typing import List, Tuple, Optional, Set, Union, Iterator
from argparse import ArgumentParser
import torch 

from separ.models.tag.tagger import Tagger
from separ.models.sdp.model import SemanticSLModel
from separ.data import SDP, EnhancedCoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc, Graph, adjacent_from_arcs
from separ.utils import SemanticMetric, flatten, pad2D, get_mask, acc, avg, Config, bar, ControlMetric

class SemanticSLParser(Tagger):
    DATASET = [SDP, EnhancedCoNLL]
    MODEL = SemanticSLModel
    DECOLLAPSE = False
    
    class Labeler:
        """Shared methods of a graph labeler."""
        SEP = '$'
        REL = 'punct'

        def __repr__(self) -> str:
            raise NotImplementedError 
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            raise NotImplementedError
        
        def decode(self, labels: List[str], rels: Optional[List[str]] = None) -> Tuple[List[Arc], bool]:
            raise NotImplementedError
        
        def recoverable(self, graph: Graph) -> List[Arc]:
            """Returns the list of arcs of the graph that are recoverable."""
            return graph.arcs 
        
        def test(self, graph: Graph) -> bool:
            """Tests the encoding, decoding and post-processing steps.

            Args:
                graph (Graph): Input graph.

            Returns:
                bool: Whether the labeler implementation is correct.
            """
            adjacent = adjacent_from_arcs(self.recoverable(graph), len(graph))
            labels, rels = self.encode(graph)
            rec, well_formed = self.decode(labels, rels)
            return (adjacent_from_arcs(rec, len(graph)) == adjacent).all() and well_formed
                
        def encode_rels(self, arcs: List[Arc], n: int) -> List[str]:
            rels = [[] for _ in range(n)]
            for arc in sorted(arcs):
                rels[arc.DEP-1].append(arc.REL)
            return [self.SEP.join(rel) for rel in rels]
        
        def decode_rels(self, arcs: List[Arc], rels: List[str]) -> List[Arc]:
            rels = [rel.split(self.SEP) for rel in rels]
            for arc in arcs:
                arc.REL = rels[arc.DEP-1].pop(0) if len(rels[arc.DEP-1]) > 0 else self.REL 
            return arcs
        
    def __init__(
        self,
        input_tkzs: List[InputTokenizer], 
        target_tkzs: List[TargetTokenizer], 
        model_confs: List[Config],
        join_rels: str,
        root_rel: str,
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.root_rel = root_rel 
        self.join_rels = join_rels
        self.LABEL = target_tkzs[0]
        
    @property
    def METRIC(self) -> SemanticMetric:
        return SemanticMetric()

    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(Tagger, cls).add_arguments(argparser)
        argparser.add_argument('--join-rels', action='store_true', help='Whether to join arc labels')
        return argparser
    
    def transform(self, graph: Graph) -> Graph:
        if not graph.transformed:
            labels, rels = self.lab.encode(graph)
            graph.__setattr__(self.LABEL.name, labels)
            if self.join_rels:
                graph.REL = rels
            else:
                graph.REL = [arc.REL for arc in sorted(graph.arcs)]
            graph.MATRIX = graph.ADJACENT
            graph.transformed = True 
        return graph
    
    def collate(self, graphs: List[Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Graph]]:
        inputs = [tkz.batch_encode(graphs, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(graphs, mode='cat', pin=False) for tkz in self.target_tkzs]
        masks = [get_mask(list(map(len, graphs)), bos=not self.join_rels)]
        if self.join_rels:
            masks.append(None)
        else:
            masks.append(pad2D([graph.MATRIX for graph in graphs]))
            masks[0][:, 0] = False
        return inputs, masks, targets, graphs
    
    def _pred(self, graph: Graph, label_pred: torch.Tensor, rel_pred: torch.Tensor) -> Tuple[Graph, bool]:
        """Predicts labeled graph.
        
        Args:
            graph (Graph): Input graph.
            label_pred (torch.Tensor): Predicted labels.

        Returns:
            Tuple[Graph, bool]: Labeled graph and well-formed indicator.
        """
        arc_pred = self.lab.decode(self.LABEL.decode(label_pred), self.REL.decode(rel_pred))
        if self.DECOLLAPSE:
            arc_pred = Graph.decollapse_one_cycles(arc_pred)
        return graph.rebuild_from_arcs(arc_pred)
        
    def _pred_rel(self, graph: Graph, arc_pred: List[Arc], rel_pred: torch.Tensor) -> Graph:
        """Assigns labels to a list of unlabeled arcs.
        
        Args: 
            graph (Graph): Input graph.
            arc_pred (List[Arc]): List of predicted arcs.
            rel_pred (torch.Tensor): Relation predictions.
            
        Returns:
            Graph: Predicted labeled graph.
        
        """
        rel_pred = self.REL.decode(rel_pred)
        for arc in arc_pred:
            arc.REL = rel_pred.pop(0) 
            if arc.HEAD == 0 and self.root_rel:
                arc.REL = self.root_rel 
        if self.DECOLLAPSE:
            arc_pred = Graph.decollapse_one_cycles(arc_pred)
        return graph.rebuild_from_arcs(arc_pred)
        
    def _pred_label(self, label_pred: torch.Tensor) -> Tuple[List[Arc], bool]:
        """Predicts unlabeled arcs.

        Args:
            label_pred (torch.Tensor): Predicted labels.

        Returns:
            Tuple[List[Arc], bool]: Predicted arcs and well-formed indicator.
        """
        return self.lab.decode(self.LABEL.decode(label_pred))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[Graph]
    ) -> Iterator[Graph]:
        """Prediction step

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[Graph]): List of input semantic graphs.

        Returns:
            Iterable[torch.Tensor]: Predicted components that are passed to self._pred.
        """
        mask, lens = masks[0], masks[0].sum(-1).tolist()
        if self.join_rels:
            label_preds, rel_preds = self.model.predict(self.model(inputs[0], inputs[1:], *masks))
            preds, _ = zip(*map(self._pred, graphs, label_preds.split(lens), rel_preds.split(lens)))
        else:
            embed = self.model.encode(*inputs)
            label_preds = self.model.predict_label(embed[mask])
            arc_preds, _ = zip(*map(self._pred_label, label_preds.split(lens)))
            matrix_preds = pad2D(map(adjacent_from_arcs, arc_preds, lens))
            num_arcs = matrix_preds.sum((-2,-1)).tolist()
            rel_preds = self.model.predict_rel(embed, matrix_preds, mask.sum()).split(num_arcs)
            preds = map(self._pred_rel, graphs, arc_preds, rel_preds)
        return preds
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        graphs: List[Graph]
    ) -> Tuple[ControlMetric, SemanticMetric]:
        mask, lens = masks[0], masks[0].sum(-1).tolist()
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        if self.join_rels:
            label_preds, rel_preds = self.model.predict(scores)
            preds, well_formed = zip(*map(self._pred, graphs, label_preds.split(lens), rel_preds.split(lens)))
        else:
            embed = self.model.encode(*inputs)
            label_preds = self.model.predict_label(embed[mask])
            arc_preds, well_formed = zip(*map(self._pred_label, label_preds.split(lens)))
            matrix_preds = pad2D(map(adjacent_from_arcs, arc_preds, lens))
            num_arcs = matrix_preds.sum((-2,-1)).tolist()
            rel_preds = self.model.predict_rel(embed, matrix_preds, mask.sum()).split(num_arcs)
            preds = map(self._pred_rel, graphs, arc_preds, rel_preds)
        return ControlMetric(
            loss=loss.detach(), well_formed=avg(well_formed)*100,
            **dict(zip(self.TARGET_FIELDS, map(acc, scores, targets)))
            ), self.METRIC(preds, graphs)
    
        
    @classmethod
    def build_inputs(
        cls,
        join_rels: bool,
        data: Union[SDP, EnhancedCoNLL],
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
    ):
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=not join_rels)]
            word_conf |= input_tkzs[0].conf
            in_confs = [word_conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM', bos=not join_rels)]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'POS', bos=not join_rels))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM', bos=not join_rels))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)
            
            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
        return input_tkzs, in_confs
    
    
    @classmethod
    def build_targets(
        cls,
        label_tkz: TargetTokenizer, 
        rel_tkz: TargetTokenizer,
        labeler: SemanticSLParser.Labeler, 
        data: Union[SDP, EnhancedCoNLL],
        join_rels: bool,
    ):
        labels, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        label_tkz.train(labels)
        if not join_rels:
            rel_tkz.train(flatten(rel.split(labeler.SEP) for rel in rels))
            rel_conf = rel_tkz.conf
            # get root rels
            root_rels = set()
            for arc in flatten(graph.arcs for graph in data):
                root_rels.add(arc.REL)
                if len(root_rels) > 1:
                    break 
            if len(root_rels) == 1:
                root_rel = root_rels.pop()
                rel_conf.special_indices.append(rel_tkz.vocab[root_rel])
            else:
                root_rel = None
        else:
            rel_tkz.train(rels)
            rel_conf = rel_tkz.conf
            root_rel = None
        rel_conf.join_rels = join_rels
        return label_tkz, rel_tkz, rel_conf, root_rel