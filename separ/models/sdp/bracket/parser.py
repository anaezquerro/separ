from __future__ import annotations
from argparse import ArgumentParser
import torch, os 
from typing import List, Tuple , Optional, Dict, Union, Iterator

from separ.data import SDP, Tokenizer, CharacterTokenizer, PretrainedTokenizer
from separ.structs import Arc, adjacent_from_arcs
from separ.utils import flatten, Config, parallel, acc, avg, pad2D, create_mask, SemanticMetric
from separ.parser import Parser
from separ.models.sdp.labeler import SemanticLabeler
from separ.models.sdp.bracket.model import BracketSemanticModel


class BracketSemanticParser(Parser):
    """Bracket Semantic Parser from [Ezquerro et al., 2024]()."""
    
    NAME = 'sdp-bracket'
    MODEL = BracketSemanticModel
    DATASET = SDP
    PARAMS = ['k']
    METRIC = SemanticMetric
    
    class Labeler(SemanticLabeler):
        """
        Sequential bracketing encoding.
        
        Examples:
        >>> labeler = BracketSemanticParser.Labeler(n=2)
        >>> graph.format()
        #20322024
        1	Volume	volume	NN	-	-	n_of:x-i	ARG1	_	_	_	_	ARG1	_	_
        2	on	on	IN	-	+	p:e-u-i	_	_	_	_	_	_	_	_
        3	the	the	DT	-	+	q:i-h-h	_	_	_	_	_	_	_	_
        4	New	_generic_proper_ne_	NNP	-	+	named:x-c	_	_	_	_	_	_	_	_
        5	York	York	NNP	-	+	named:x-c	_	_	compound	_	_	_	_	_
        6	Stock	stock	NNP	-	+	n:x	_	_	_	_	_	_	_	_
        7	Exchange	exchange	NNP	-	-	n:x	ARG2	BV	_	compound	compound	_	_	_
        8	totaled	total	VBD	+	+	v:e-i-i	_	_	_	_	_	_	_	_
        9	176.1	_generic_card_ne_	CD	-	+	card:i-i-c	_	_	_	_	_	_	_	_
        10	million	million	CD	-	+	card:i-i-c	_	_	_	_	_	_	times	_
        11	shares	share	NNS	-	-	n_of:x	_	_	_	_	_	ARG2	_	ARG1
        12	.	_	.	-	-	_	_	_	_	_	_	_	_	_
        >>> graph.arcs
        [
            2 --(ARG1)--> 1,
            8 --(ARG1)--> 1,
            4 --(compound)--> 5,
            2 --(ARG2)--> 7,
            3 --(BV)--> 7,
            5 --(compound)--> 7,
            6 --(compound)--> 7,
            0 --(TOP)--> 8,
            9 --(times)--> 10,
            8 --(ARG2)--> 11,
            10 --(ARG1)--> 11
        ]
        >>> brackets = labeler.encode(graph)
        >>> brackets
        ['/', '<<', '\\/', '/', '/', '>/', '/', '>>>>', '\\>/', '/', '>/', '>>', '']
        >>> labeler.decode(brackets)
        [
            2 --(None)--> 1,
            4 --(None)--> 5,
            6 --(None)--> 7,
            5 --(None)--> 7,
            3 --(None)--> 7,
            2 --(None)--> 7,
            0 --(None)--> 8,
            8 --(None)--> 1,
            9 --(None)--> 10,
            10 --(None)--> 11,
            8 --(None)--> 11
        ]
        """
        BRACKETS = [ '\\', '>', '<', '/']
        DEFAULT = ''
        
        def __init__(self, k: int):
            self.k = k 
            
        def __repr__(self) -> str:
            return f'BracketSemanticLabeler(k={self.k})'
            
        def split_bracket(self, bracket: str) -> List[str]:
            """Splits a bracket token label into different brackets.
            
            Args:
                bracket (str): Bracket token label.

            Returns:
                List[str]: List of brackets that conform the token label.
                
            Examples:
            >>> labeler.split_bracket('\\/>')
            ['\\', '/', '>']
            """
            stack = []
            for x in bracket:
                if x == '*':
                    stack[-1] += x 
                else:
                    stack.append(x)
            return stack 
        
        def preprocess(self, graph: SDP.Graph) -> List[List[Arc]]:
            return [graph.planes[p] for p in range(min(self.k, len(graph.planes)))]
        
        def encode(self, graph: SDP.Graph) -> List[str]:
            """Encodes and input semantic graph.

            Args:
                graph (SDP.Graph): Input semantic graph.

            Returns:
                List[str]: k-planar bracketing encoding.
                
            Examples:
                >>> print(graph.format())
                #20018026
                1	Cray	_generic_proper_ne_	NNP	-	+	named:x-c	_	_	_	_
                2	Computer	_generic_proper_ne_	NNP	-	-	named:x-c	compound	ARG1	ARG1	_
                3	has	has	VBZ	-	-	_	_	_	_	_
                4	applied	apply	VBN	+	+	v:e-i-h	_	_	_	_
                5	to	to	TO	-	-	_	_	_	_	_
                6	trade	trade	VB	-	+	v:e-i-p	_	ARG2	_	ARG1
                7	on	on	IN	-	+	p:e-u-i	_	_	_	_
                8	Nasdaq	_generic_proper_ne_	NNP	-	-	named:x-c	_	_	_	ARG2
                9	.	_	.	-	-	_	_	_	_	_
                >>> graph.arcs
                [
                    1 --(compound)--> 2,
                    4 --(ARG1)--> 2,
                    6 --(ARG1)--> 2,
                    0 --(TOP)--> 4,
                    4 --(ARG2)--> 6,
                    7 --(ARG1)--> 6,
                    7 --(ARG2)--> 8
                ]
                >>> len(graph.planes)
                2
                >>> BracketSemanticParser.Labeler(k=2).encode(graph)
                ['/', '><<*', '', '\\>/', '', '\\*><', '\\/', '>', '']
            """
            k = min(self.k, len(graph.planes))
            count = [self._encode(graph.planes[p], len(graph)) for p in range(k)]
            brackets = []
            for idep in range(len(graph)):
                bracket = ''
                for br in self.BRACKETS:
                    for p in range(k):
                        bracket += (br+'*'*p)*count[p][idep][br] 
                brackets.append(bracket)
            return brackets
              
        def decode(self, brackets: List[str]) -> List[Arc]:
            """Decodes a sequence of k-planar brackets.

            Args:
                brackets (List[str]): Sequence of k-planar brackets.

            Returns:
                List[Arc]: Decoded arcs.
                
            Examples:
                >>> lab.decode(['/', '><<*', '', '\\>/', '', '\\*><', '\\/', '>', ''])
                [
                    1 --(None)--> 2,
                    4 --(None)--> 2,
                    6 --(None)--> 2,
                    0 --(None)--> 4,
                    4 --(None)--> 6,
                    7 --(None)--> 6,
                    7 --(None)--> 8
                ]
            """
            planes = [[[] for _ in range(len(brackets))] for _ in range(self.k)]
            for i, bracket in enumerate(brackets):
                for br in self.split_bracket(bracket):
                    planes[br.count('*')][i].append(br)
            return sorted(flatten(map(self._decode, planes)))
                   
        def decode_postprocess(self, brackets: List[str]) -> Tuple[List[Arc], torch.Tensor, bool]:
            planes = [[[] for _ in range(len(brackets))] for _ in range(self.k)]
            for i, bracket in enumerate(brackets):
                for br in self.split_bracket(bracket):
                    planes[br.count('*')][i].append(br)
            arcs, well_formed = zip(*map(self._decode_postprocess, planes))
            arcs = sorted(set(flatten(arcs)))
            return arcs, adjacent_from_arcs(arcs, len(brackets)), all(well_formed)
        
        def _encode(self, plane: List[Arc], n: int) -> List[Dict[str, int]]:
            """Represents a plane using the bracketing-encodings.

            Args:
                plane (List[Arc]): Input plane (non-crossing arcs).
                n (int): Number of nodes in the graph.

            Returns:
                List[Dict[str, int]]: Number of bracket symbols per token.
            """
            # dictionary to store the count of each bracket
            count = [{br: 0 for br in self.BRACKETS} for _ in range(n)]
            for arc in plane:
                if arc.DEP < arc.HEAD: # left arc 
                    count[arc.HEAD-1]['\\'] += 1
                    count[arc.DEP-1]['<'] += 1
                else: # right arc 
                    if arc.HEAD > 0:
                        count[arc.HEAD-1]['/'] += 1
                    count[arc.DEP-1]['>'] += 1 
            return count 

        def _decode(self, brackets: List[List[str]]) -> List[Arc]:
            """Decodes a sequence of 1-planar brackets.

            Args:
                brackets (List[str]): Input sequence of 1-planar brackets.

            Returns:
                List[Arc]: List of decoded arcs.
            """
            right, left, arcs = [0], [], []
            for idep, bracket in enumerate(brackets):
                dep = idep+1
                for b in bracket:
                    if '>' in b:
                        arcs.append(Arc(right[-1], dep, None))
                        if right[-1] != 0:
                            right.pop(-1)
                    elif '/' in b:
                        right.append(dep)
                    elif '\\' in b:
                        arcs.append(Arc(dep, left.pop(-1), None))
                    elif '<' in b:
                        left.append(dep)
            return arcs
        
        def _decode_postprocess(self, brackets: List[List[str]]) -> Tuple[List[Arc], bool]:
            """Decodes and post-processes a sequence of 1-planar brackets.

            Args:
                brackets (List[str]): Input sequence of 1-planar brackets.

            Returns:
                List[Arc]: List of decoded arcs and boolean value to check the correctness of the 
                    bracket sequence.
            """
            right, left, arcs = [0], [], []
            for idep, bracket in enumerate(brackets):
                dep = idep+1
                for b in bracket:
                    if '>' in b:
                        arcs.append(Arc(right[-1], dep, None))
                        if right[-1] != 0:
                            right.pop(-1)
                    elif '/' in b:
                        right.append(dep)
                    elif '\\' in b and len(left) > 0:
                        arcs.append(Arc(dep, left.pop(-1), None))
                    elif '<' in b:
                        left.append(dep)
            right.pop(0) # pop the node w0
            # the sequence of brackets are well-formed if the both stacks are empty
            return arcs, len(left) == len(right) == 0
        
    def __init__(
        self,
        model: BracketSemanticModel, 
        input_tkzs: List[Union[Tokenizer, CharacterTokenizer, PretrainedTokenizer]],
        target_tkzs: List[Tokenizer],
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
        argparser.add_argument('-k', type=int, help='Number of bracket planes')
        return argparser 
    
    @classmethod
    def transform(
        cls, 
        graph: SDP.Graph, 
        input_tkzs: List[Tokenizer], 
        BRACKET: Tokenizer, 
        REL: Tokenizer, 
        labeler: BracketSemanticParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            graph.__setattr__(BRACKET.name, BRACKET.encode(labeler.encode(graph)).pin_memory())
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
        brackets, rels, matrices, mask = *targets, *masks
        s_bracket, s_rel = self.model(inputs[0], inputs[1:], matrices, mask)
        loss = self.model.loss(s_bracket, s_rel, brackets, rels)
        return loss, dict(zip(self.TARGET_FIELDS, map(acc, (s_bracket, s_rel), (brackets, rels))))
    
    @torch.no_grad()
    def pred_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[SDP.Graph]
    ) -> Iterator[SDP.Graph]:
        mask, lens = masks[0], masks[0].sum(-1).tolist()
        embed = self.model.encode(*inputs)
        bracket_preds = self.model.bracket_pred(embed, mask)
        arc_preds, matrix_preds, _ = zip(*map(self.labeler.decode_postprocess, map(self.BRACKET.decode, bracket_preds.split(lens))))
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
        brackets, rels, _, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, embed, bracket_preds, s_rel = self.model.control(inputs[0], inputs[1:], *targets, mask)
        arc_preds, matrix_preds, well_formed = zip(*map(self.labeler.decode_postprocess, map(self.BRACKET.decode, bracket_preds.split(lens))))
        matrix_preds = pad2D(matrix_preds)[:, 1:, 1:] # suppress arcs to head
        rel_preds = self.model.rel_pred(embed, matrix_preds, mask.sum()).split(matrix_preds.sum((-2,-1)).tolist())
        control = dict(BRACKET=acc(bracket_preds, brackets), REL=acc(s_rel, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
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
    ) -> BracketSemanticParser:
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
                
        bracket_tkz, rel_tkz = Tokenizer('BRACKET'), Tokenizer('REL')
        labeler = cls.Labeler(k)
        if pretest:
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        bracket_tkz.train(*flatten(parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        rel_tkz.train(*[arc.REL for graph in data for arc in graph.arcs])
        
        model = cls.MODEL(enc_conf, *in_confs, bracket_tkz.conf, rel_tkz.conf).to(device)
        return cls(model, input_tkzs, [bracket_tkz, rel_tkz], k, device)

