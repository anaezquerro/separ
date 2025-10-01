from __future__ import annotations
from argparse import ArgumentParser
import torch 

from separ.models.grp.parser import GraphParser
from separ.models.tag.tagger import Tagger 
from separ.models.grp.model import GraphSLModel
from separ.data import SDP, EnhancedCoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, Arc, Graph, adjacent_from_arcs
from separ.utils import GraphMetric, flatten, pad2D, get_mask, avg, Config, bar, ControlMetric, TaggingMetric

class GraphSLParser(GraphParser, Tagger):
    MODEL = GraphSLModel
    DECOLLAPSE = False
    
    class Labeler:
        """Shared methods of a graph labeler."""
        SEP = '$'
        REL = 'punct'
        PARAMS = set()

        def __repr__(self) -> str:
            raise NotImplementedError 
        
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            raise NotImplementedError
        
        def decode(self, labels: list[str], rels: list[str] | None = None) -> tuple[list[Arc], bool]:
            raise NotImplementedError
        
        def recoverable(self, graph: Graph) -> list[Arc]:
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
                
        def encode_rels(self, arcs: list[Arc], n: int) -> list[str]:
            rels = [[] for _ in range(n)]
            for arc in sorted(arcs):
                rels[arc.DEP-1].append(arc.REL)
            return [self.SEP.join(rel) for rel in rels]
        
        def decode_rels(self, arcs: list[Arc], rels: list[str]) -> list[Arc]:
            rels = [rel.split(self.SEP) for rel in rels]
            for arc in arcs:
                arc.REL = rels[arc.DEP-1].pop(0) if len(rels[arc.DEP-1]) > 0 else self.REL 
            return arcs
        
    def __init__(self, **_):
        super().__init__(**_)
        self.lab = self.Labeler(**_)
        
    @property 
    def PARAMS(self) -> set[str]:
        return GraphParser.PARAMS | {'join_rels'} | self.Labeler.PARAMS
        
    @property
    def LABEL(self) -> TargetTokenizer:
        return self.target_tkzs[0]
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = GraphParser.add_arguments(argparser)
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
    
    def collate(self, graphs: list[Graph]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[Graph]]:
        inputs = [tkz.batch_encode(graphs, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(graphs, mode='cat', pin=False) for tkz in self.target_tkzs]
        if self.join_rels:
            mask = get_mask(list(map(len, graphs)))
            masks = [mask, mask]
        else:
            mask = get_mask(list(map(len, graphs)), bos=True)
            mask[:, 0] = False
            masks = [mask, pad2D([graph.MATRIX for graph in graphs])]
        return inputs, masks, targets, graphs
    
    def _pred(self, graph: Graph, label_pred: torch.Tensor, rel_pred: torch.Tensor) -> tuple[Graph, bool]:
        """Predicts labeled graph.
        
        Args:
            graph (Graph): Input graph.
            label_pred (torch.Tensor): Predicted labels.

        Returns:
            tuple[Graph, bool]: Labeled graph and well-formed indicator.
        """
        arc_pred, well_formed = self.lab.decode(self.LABEL.decode(label_pred), self.REL.decode(rel_pred))
        if self.DECOLLAPSE:
            arc_pred = Graph.decollapse_one_cycles(arc_pred)
        return graph.rebuild_from_arcs(arc_pred), well_formed
        
    def _pred_rel(self, graph: Graph, arc_pred: list[Arc], rel_pred: torch.Tensor) -> Graph:
        """Assigns labels to a list of unlabeled arcs.
        
        Args: 
            graph (Graph): Input graph.
            arc_pred (list[Arc]): List of predicted arcs.
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
        
    def _pred_label(self, label_pred: torch.Tensor) -> tuple[list[Arc], bool]:
        """Predicts unlabeled arcs.

        Args:
            label_pred (torch.Tensor): Predicted labels.

        Returns:
            tuple[list[Arc], bool]: Predicted arcs and well-formed indicator.
        """
        return self.lab.decode(self.LABEL.decode(label_pred))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        graphs: list[Graph]
    ) -> list[Graph]:
        """Prediction step

        Args:
            inputs (list[torch.Tensor]): List of batched and padded inputs.
            masks (list[torch.Tensor]): List of padding masks.
            graphs (list[Graph]): List of input semantic graphs.

        Returns:
            Iterable[torch.Tensor]: Predicted components that are passed to self._pred.
        """
        return self._pred_step(inputs, masks, graphs)[0]
    
    @torch.no_grad()
    def _pred_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        graphs: list[Graph]
    ) -> tuple[list[Graph], float]:
        if self.join_rels:
            preds = self.model.predict(self.model(inputs[0], inputs[1:], *masks))
            preds = [pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]
            preds, well_formed = zip(*map(self._pred, graphs, *preds))
        else:
            mask, _ = masks
            lens = mask.sum(-1).tolist()
            embed = self.model.encode(*inputs)
            label_preds = self.model.predict_label(embed[mask])
            arc_preds, well_formed = zip(*map(self._pred_label, label_preds.split(lens)))
            matrix_preds = pad2D(map(adjacent_from_arcs, arc_preds, lens))
            num_arcs = matrix_preds.sum((-2,-1)).tolist()
            rel_preds = self.model.predict_rel(embed, matrix_preds, mask.sum()).split(num_arcs)
            preds = map(self._pred_rel, graphs, arc_preds, rel_preds)
        return list(preds), avg(well_formed)*100
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        graphs: list[Graph]
    ) -> tuple[ControlMetric, GraphMetric, list[Graph]]:
        _, control = self.train_step(inputs, masks, targets)
        preds, well_formed = self._pred_step(inputs, masks, graphs)
        control |= ControlMetric(well_formed=well_formed)
        return control, self.METRIC(preds, graphs), preds
        
    @classmethod
    def build_inputs(
        cls,
        join_rels: bool,
        data: SDP | EnhancedCoNLL,
        word_conf: Config, 
        tag_conf: Config | None,
        char_conf: Config | None,
    ):
        if hasattr(word_conf, 'pretrained'):
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
        lab: GraphSLParser.Labeler, 
        data: SDP | EnhancedCoNLL,
        join_rels: bool,
    ) -> tuple[list[TargetTokenizer], list[Config], bool]:
        target_tkzs = [TargetTokenizer(name) for name in lab.NAMES]
        *labels, rels = map(flatten, zip(*bar(map(lab.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        for tkz, _labels in zip(target_tkzs, labels):
            tkz.train(_labels)
        
        rel_tkz = target_tkzs[-1]
        if not join_rels:
            rel_tkz.train(flatten(rel.split(lab.SEP) for rel in rels))
            rel_conf = rel_tkz.conf
            # get root rels
            root_rels = set()
            for arc in flatten(graph.arcs for graph in data):
                if arc.HEAD == 0:
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
        return target_tkzs, [tkz.conf for tkz in target_tkzs[:-1]] + [rel_conf], root_rel
    
    @classmethod
    def build(
        cls,
        data: SDP | EnhancedCoNLL | str, 
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        join_rels: bool = False,
        **_
    ) -> GraphSLParser:
        if isinstance(data, str):
            data = cls.load_data(data)
        
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        target_tkzs, target_confs, root_rel = cls.build_targets(cls.Labeler(**_), data, join_rels)
        return cls(
            input_tkzs=input_tkzs,
            target_tkzs=target_tkzs, 
            model_confs=[enc_conf, *in_confs] + target_confs, 
            join_rels=join_rels,
            root_rel=root_rel,
            **_
        )
        