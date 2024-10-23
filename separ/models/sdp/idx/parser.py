from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Union, Dict, Iterator
import torch

from separ.data import SDP, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.structs import Arc, adjacent_from_arcs
from separ.utils import flatten, Config, parallel, SemanticMetric, pad2D, create_mask, acc, avg
from separ.parser import Parser
from separ.models.sdp.idx.model import IndexSemanticModel
from separ.models.sdp.labeler import SemanticLabeler


class IndexSemanticParser(Parser):
    """Index Semantic Parser from [Ezquerro et al., 2024]().

    Inherits:
        ``SLSemanticParser``: train_step, control_step, pred_step, _pred, collate, transform.
        
    Methods: init, Labeler, add_arguments, build.
    """
    NAME = 'sdp-idx'
    MODEL = IndexSemanticModel
    DATASET = SDP
    METRIC = SemanticMetric
    PARAMS = ['rel']
    
    class Labeler(SemanticLabeler):
        SEP = '$'
        DEFAULT = ''
        
        def __init__(self, rel: bool = False):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexSemanticLabeler(rel={self.rel})'
            
        def encode(self, graph: SDP.Graph) -> List[str]:
            indexes = [[] for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1].append(arc.HEAD - (arc.DEP*self.rel))
            indexes = ['' if len(index) == 0 else self.SEP.join(map(str, sorted(index))) for index in indexes]
            return indexes
        
        def decode(self, indexes: List[str]) -> List[Arc]:
            arcs = []
            for idep, index in enumerate(indexes):
                if index == '':
                    continue 
                for idx in index.split(self.SEP):
                    head = int(idx)+(idep+1)*self.rel 
                    arcs.append(Arc(head, idep+1, None))
            return sorted(arcs)
        
        def decode_postprocess(self, indexes: List[str]) -> Tuple[List[Arc], torch.Tensor, bool]:
            n, arcs, well_formed = len(indexes), [], True
            for idep, index in enumerate(indexes):
                if index == '':
                    continue 
                for idx in index.split(self.SEP):
                    head = int(idx)+(idep+1)*self.rel 
                    if head in range(n+1) and head != (idep+1):
                        arcs.append(Arc(head, idep+1, None))
                    else:
                        well_formed = False
            return sorted(arcs), adjacent_from_arcs(arcs, len(indexes)), well_formed
        
    def __init__(
        self, 
        model: IndexSemanticModel, 
        input_tkzs: List[Union[Tokenizer, PretrainedTokenizer, CharacterTokenizer]], 
        target_tkzs: List[Tokenizer], 
        rel: bool,
        device: Union[int, str]
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.rel = rel
        self.labeler = self.Labeler(rel)
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('--rel', action='store_true', help='Relative indexing')
        return argparser 
    
    @classmethod
    def transform(
        cls, 
        graph: SDP.Graph, 
        input_tkzs: List[Tokenizer], 
        INDEX: Tokenizer, 
        REL: Tokenizer, 
        labeler: IndexSemanticParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            graph.__setattr__(INDEX.name, INDEX.encode(labeler.encode(graph)).pin_memory())
            graph.REL = REL.encode([arc.REL for arc in sorted(graph.arcs) if arc.HEAD > 0]).pin_memory()
            graph.MATRIX = graph.ADJACENT[1:, 1:].pin_memory()
            graph._transformed = True 
    
    def collate(self, batch: List[SDP.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[SDP.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in batch]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in batch]) for tkz in self.target_tkzs]
        targets.append(pad2D([graph.MATRIX for graph in batch]))
        masks = [create_mask(list(map(len, batch)))]
        return inputs, targets, masks, batch
        
    def _pred(self, graph: SDP.Graph, arc_pred: List[Arc], rel_pred: torch.Tensor) -> SDP.Graph:
        rel_pred = self.REL.decode(rel_pred)
        for arc in arc_pred:
            if arc.HEAD == 0:
                arc.REL = 'root'
            else:
                arc.REL = rel_pred.pop(0)
        return graph.rebuild(arc_pred)
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and padded/concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Loss and control metrics.
        """
        indexes, rels, matrices, mask = *targets, *masks
        s_index, s_rel = self.model(inputs[0], inputs[1:], matrices, mask)
        loss = self.model.loss(s_index, s_rel, indexes, rels)
        return loss, dict(INDEX=acc(s_index, indexes), REL=acc(s_rel, rels))
        
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[SDP.Graph]
    ) -> Iterator[SDP.Graph]:
        """Prediction step:
            1. Predict the ``indexes`` from the inputs (1st component).
            2. Decode to recover the arcs.
            3. Predict the relations from the head and dependant embeddings.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[SDP.Graph]): List of input semantic graphs.

        Returns:
            Iterator[SDP.Graph]: Predicted semantic graphs.
        """
        mask, lens = masks[0], masks[0].sum(-1).tolist()
        embed = self.model.encode(*inputs)
        index_preds = self.model.index_pred(embed, mask)
        arc_preds, matrix_preds, _ = zip(*map(self.labeler.decode_postprocess, map(self.INDEX.decode, index_preds.split(lens))))
        matrix_preds = pad2D(matrix_preds)[:, 1:, 1:] # suppress arcs to head
        rel_preds = self.model.rel_pred(embed, matrix_preds, mask.sum()).split(matrix_preds.sum((-2,-1)).tolist())
        return map(self._pred, graphs, arc_preds, rel_preds)
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[SDP.Graph]
    ) -> Tuple[Dict[str, float], Iterator[SDP.Graph]]:
        """Control (prediction + evaluation) step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and padded/concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[SDP.Graph]): List of input semantic graphs.

        Returns: 
            Tuple[Dict[str, float], Iterator[SDP.Graph]]: Control metrics and predicted semantic graphs.
        """
        indexes, rels, _, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, embed, index_preds, s_rel = self.model.control(inputs[0], inputs[1:], *targets, mask)
        arc_preds, matrix_preds, well_formed = zip(*map(self.labeler.decode_postprocess, map(self.INDEX.decode, index_preds.split(lens))))
        matrix_preds = pad2D(matrix_preds)[:, 1:, 1:] # suppress arcs to head
        rel_preds = self.model.rel_pred(embed, matrix_preds, mask.sum()).split(matrix_preds.sum((-2,-1)).tolist())
        control = dict(INDEX=acc(index_preds, indexes), REL=acc(s_rel, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
        return control, map(self._pred, graphs, arc_preds, rel_preds)
    
    @classmethod 
    def build(
        cls, 
        data: Union[SDP, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        rel: bool = False,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> IndexSemanticParser:
        if isinstance(data, str):
            data = SDP.from_file(data, num_workers)
        
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
                conf.join(tkz.conf)
        
        # train target tokenizers 
        index_tkz, rel_tkz = Tokenizer('INDEX'), Tokenizer('REL')
        labeler = cls.Labeler(rel=rel)
        if pretest:
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        index_tkz.train(*flatten(parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        rel_tkz.train(*[arc.REL for graph in data for arc in graph.arcs])
            
        model = cls.MODEL(enc_conf, *in_confs, index_tkz.conf, rel_tkz.conf).to(device)
        return cls(model, input_tkzs, [index_tkz, rel_tkz], rel, device)
        