from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Optional, Union, Tuple, Dict, Iterator
import torch, os 

from separ.data import SDP, Tokenizer, PretrainedTokenizer, CharacterTokenizer, AbstractTokenizer
from separ.structs import Arc, adjacent_from_arcs
from separ.utils import Config, split, parallel, flatten, pad2D, create_mask, acc, avg, SemanticMetric
from separ.parser import Parser
from separ.models.sdp.labeler import SemanticLabeler
from separ.models.sdp.bit6k.model import Bit6kSemanticModel


class Bit6kSemanticParser(Parser):
    """6k-bit Semantic Parser from [Ezquerro et al., 2024]()."""
    NAME = 'sdp-bit6k'
    MODEL = Bit6kSemanticModel
    DATASET = SDP
    METRIC = SemanticMetric
    PARAMS = ['k']
    
    class Labeler(SemanticLabeler):
        """6k-bit encoding.
        - b0: word is a left dependant in a plane.
        - b1: word is a right dependant in a plane.
        - b2: word is the farthest left dependant in a plane.
        - b3: word is the farthest right dependant in a plane.
        - b4: word has left dependants in a plane.
        - b5: word has right dependants in a plane.
        """
        N_BITS = 6
        
        def __init__(self, k: int = 3):
            self.k = k 
            
        def __repr__(self) -> str:
            return f'Bit6kSemanticLabeler(k={self.k})'
        
        @property
        def DEFAULT(self) -> str:
            return '000000'*self.k
            
        def preprocess(self, graph: SDP.Graph) -> List[List[Arc]]:
            """Divides the graph in a set of 6k-bit planes. Two arcs cannot belong to the same 
            plane if:
                1. They cross each other in the same direction.
                2. They share the same dependant in the same direction.

            Args:
                graph (SDP.Graph): Input semantic graph.

            Returns:
                List[List[Arc]]: List of k planes.
            """
            if not graph.bit6k_planes:
                planes = [[]]
                adjacent = [adjacent_from_arcs([], len(graph))]
                for arc in sorted(graph.arcs, key=lambda arc: arc.left):
                    added = False 
                    for p in range(len(planes)):
                        # the arc cannot cross any other arc in the same direction 
                        # there cannot be another arc with the same dependant and a head in the same direction
                        if not any(arc.cross(other) and arc.side == other.side for other in planes[p]) \
                            and not (adjacent[p][arc.DEP, arc.DEP+1:] if arc.side == -1 else adjacent[p][arc.DEP, :arc.DEP]).any():
                            planes[p].append(arc)
                            adjacent[p][arc.DEP, arc.HEAD] = True 
                            added = True 
                            break 
                    # if not added, add a new plane 
                    if not added:
                        planes.append([arc])
                        adjacent.append(adjacent_from_arcs(planes[-1], len(graph)))
                
                # check if the number of planes is less than k
                if len(planes) < self.k:
                    # add empty planes
                    planes += [[] for _ in range(len(planes), self.k)]
                elif len(planes) > self.k:
                    # sort planes by the number of arcs or the index
                    order = sorted(range(len(planes)), key=lambda i: (len(planes[i]), -i), reverse=True)
                    planes = [planes[p] for p in order[:self.k]]
                graph.bit6k_planes = planes 
            return graph.bit6k_planes 
            
        def encode(self, graph: SDP.Graph) -> List[str]:
            """Encodes a semantic graph with the 6k-bit representation.

            Args:
                graph (SDP.Graph): Input semantic graph.

            Returns:
                List[str]: Sequence of 6k-bit labels.
            """
            planes = self.preprocess(graph)
            n = len(graph)
            labels = []
            for label in zip(*[self._encode(planes[p], n) for p in range(self.k)]):
                labels.append(''.join(label))
            return labels 
        
        def decode(self, labels: List[str]) -> List[Arc]:
            """Decodes the 6k-bit representation for a full sequence of labels.

            Args:
                labels (List[str]): 6k-bit input sequence.

            Returns:
                List[Arc]: Decoded arcs.
            """
            planes = zip(*[[''.join(lab) for lab in split(list(label), self.N_BITS)] for label in labels])
            return sorted(set(flatten(map(self._decode, planes))))
        
        def decode_postprocess(self, labels: List[str]) -> Tuple[List[Arc], torch.Tensor, bool]:
            """Decodes the 6k-bit representation for a full sequence of labels.

            Args:
                labels (List[str]): 6k-bit input sequence.

            Returns:
                Tuple[List[Arc], torch.Tensor, bool]: Decoded arcs, adjacent matrix and whether the 
                    input sequence produces a well-formed semantic graph.
            """
            planes = zip(*[[''.join(lab) for lab in split(list(label), self.N_BITS)] for label in labels])
            arcs, well_formed = zip(*map(self._decode_postprocess, planes))
            arcs = sorted(set(flatten(arcs)))
            return arcs, adjacent_from_arcs(arcs, len(labels)), all(well_formed)
            
        def _encode(self, plane: List[Arc], n: int) -> List[str]:
            """Represents a plane with the 6-bit encoding.

            Args:
                plane (List[Arc]): List of non-crossing arcs.
                n (int): Number of nodes in the graph.
                
            Returns:
                List[str]: 6-bit representation.
            """
            labels = [[False for _ in range(self.N_BITS)] for _ in range(n)]
            adjacent = adjacent_from_arcs(plane, n)
            for arc in plane:
                if arc.DEP < arc.HEAD: # left arc 
                    # b0: DEP is a left dependant in the plane 
                    labels[arc.DEP-1][0] = True 
                    # b2: DEP is the farthest left dependant in the plane
                    labels[arc.DEP-1][2] = not adjacent[:arc.DEP, arc.HEAD].any()
                    # b4: HEAD has left dependants in the plane 
                    labels[arc.HEAD-1][4] = True 
                else: # right arc 
                    # b1: DEP is a right dependant in the plane 
                    labels[arc.DEP-1][1] = True 
                    # b3: DEP is the farthest right dependant in the plane 
                    labels[arc.DEP-1][3] = not adjacent[(arc.DEP+1):, arc.HEAD].any()
                    # b5: HEAD has right dependants in the plane 
                    if arc.HEAD != 0:
                        labels[arc.HEAD-1][5] = True 
            return [''.join(str(int(bit)) for bit in label) for label in labels]
                
        def _decode(self, labels: List[str]) -> List[Arc]:
            """Decodes the 6-bit representation for a given plane.

            Args:
                labels (List[str]): 6-bit input representation.

            Returns:
                List[Arc]: Decoded non-crossing arcs.
            """
            right, left, arcs = [0], [], []
            for idep, label in enumerate(labels):
                dep = idep+1
                b0, b1, b2, b3, b4, b5 = map(bool, map(int, label)) 
                if b1: # DEP is a right dependant in the plane
                    arcs.append(Arc(right[-1], dep, None))
                    if b3 and right[-1] != 0: # DEP is the farthest right dependant in the plane
                        right.pop(-1)

                if b4: # DEP has left dependants in the plane 
                    last = False 
                    while not last:
                        last = left[-1][-1]
                        arcs.append(Arc(dep, left.pop(-1)[0], None))
                        
                if b5: # DEP has right dependants in the plane 
                    right.append(dep)
                    
                if b0: # DEP is a left dependant in the plane 
                    left.append((dep, b2))
            return arcs 
        
        def _decode_postprocess(self, labels: List[str]) -> Tuple[List[Arc], bool]:
            right, left, arcs = [0], [], []
            for idep, label in enumerate(labels):
                dep = idep+1
                label = map(bool, map(int, label))
                b0, b1, b2, b3, b4, b5 = label 
                if b1 and len(right) > 0: # DEP is a right dependant in the plane
                    arcs.append(Arc(right[-1], dep, None))
                    if b3 and right[-1] != 0: # DEP is the farthest right dependant in the plane
                        right.pop(-1)

                if b4: # DEP has left dependants in the plane 
                    last = False 
                    while not last and len(left) > 0:
                        last = left[-1][-1]
                        arcs.append(Arc(dep, left.pop(-1)[0], None))
                        
                if b5: # DEP has right dependants in the plane 
                    right.append(dep)
                    
                if b0: # DEP is a left dependant in the plane 
                    left.append((dep, b2))
            right.pop(0) # pop the node w0
            return arcs, len(right) == len(left) == 0
            
    def __init__(
        self,
        model: Bit6kSemanticModel, 
        input_tkzs: List[AbstractTokenizer], 
        target_tkzs: List[AbstractTokenizer],
        k: int,
        device: str
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.k = k
        self.labeler = self.Labeler(k)
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('-k', type=int, default=3, help='Number of planes for the 6-bit encoding')
        return argparser 
    
    @classmethod
    def transform(
        cls, 
        graph: SDP.Graph, 
        input_tkzs: List[Tokenizer], 
        BIT: Tokenizer, 
        REL: Tokenizer, 
        labeler: Bit6kSemanticParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            graph.__setattr__(BIT.name, BIT.encode(labeler.encode(graph)).pin_memory())
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
        bits, rels, matrices, mask = *targets, *masks
        s_bit, s_rel = self.model(inputs[0], inputs[1:], matrices, mask)
        loss = self.model.loss(s_bit, s_rel, bits, rels)
        return loss, dict(zip(self.TARGET_FIELDS, map(acc, (s_bit, s_rel), (bits, rels))))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[SDP.Graph]
    ) -> Iterator[SDP.Graph]:
        mask, lens = masks[0], masks[0].sum(-1).tolist()
        embed = self.model.encode(*inputs)
        bit_preds = self.model.bit_pred(embed, mask)
        arc_preds, matrix_preds, _ = zip(*map(self.labeler.decode_postprocess, map(self.BIT.decode, bit_preds.split(lens))))
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
        bits, rels, _, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, embed, bit_preds, s_rel = self.model.control(inputs[0], inputs[1:], *targets, mask)
        arc_preds, matrix_preds, well_formed = zip(*map(self.labeler.decode_postprocess, map(self.BIT.decode, bit_preds.split(lens))))
        matrix_preds = pad2D(matrix_preds)[:, 1:, 1:] # suppress arcs to head
        rel_preds = self.model.rel_pred(embed, matrix_preds, mask.sum()).split(matrix_preds.sum((-2,-1)).tolist())
        control = dict(BIT=acc(bit_preds, bits), REL=acc(s_rel, rels), loss=loss, well_formed=avg(well_formed)*100)
        return control, map(self._pred, graphs, arc_preds, rel_preds)
    
    @classmethod
    def build(
        cls,
        data: Union[SDP, str], 
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        k: int = 2,
        device: str = 'cuda:0',
        pretest: bool = False,
        num_workers: int = os.cpu_count(),
        **_
    ) -> Bit6kSemanticParser:
        if isinstance(data, str):
            data = SDP.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            in_confs = [word_conf | input_tkzs[0].conf, None, None]
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
                
        bit_tkz, rel_tkz = Tokenizer('BIT'), Tokenizer('REL')
        labeler = cls.Labeler(k)
        if pretest:
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        rel_tkz.train(*[arc.REL for graph in data for arc in graph.arcs])
        bit_tkz.train(*flatten(parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        
        model = cls.MODEL(enc_conf, *in_confs, bit_tkz.conf, rel_tkz.conf).to(device)
        return cls(model, input_tkzs, [bit_tkz, rel_tkz], k, device=device)
        