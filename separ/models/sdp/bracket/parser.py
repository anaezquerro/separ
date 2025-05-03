from __future__ import annotations
from argparse import ArgumentParser
import torch, os 
from typing import List, Tuple , Optional, Dict, Union, Optional

from separ.data import InputTokenizer, TargetTokenizer, SDP, EnhancedCoNLL, Arc, Graph
from separ.utils import flatten, Config
from separ.models.sdp.parser import SemanticSLParser


class BracketSemanticParser(SemanticSLParser):
    """Bracket Semantic Parser from [Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)."""
    
    NAME = 'sdp-bracket'
    PARAMS = ['k', 'join_rels', 'root_rel']
    DECOLLAPSE = True 
    
    class Labeler(SemanticSLParser.Labeler):
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
        
        def recoverable(self, graph: Graph) -> List[Arc]:
            return flatten(graph.relaxed_planes[p] for p in range(min(self.k, len(graph.relaxed_planes))))
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            k = min(self.k, len(graph.relaxed_planes))
            planes = {p: graph.relaxed_planes[p] for p in range(k)}
            count = [self._encode(planes[p], len(graph)) for p in range(k)]
            brackets = []
            for idep in range(len(graph)):
                bracket = ''
                for br in self.BRACKETS:
                    for p in range(k):
                        bracket += (br+'*'*p)*count[p][idep][br] 
                brackets.append(bracket)
            return brackets, self.encode_rels(flatten(*planes.values()), len(graph))
              
        def decode(self, brackets: List[str], rels: Optional[List[str]] = None) -> Tuple[List[Arc], bool]:
            planes = [[[] for _ in range(len(brackets))] for _ in range(self.k)]
            for i, bracket in enumerate(brackets):
                for br in self.split_bracket(bracket):
                    planes[br.count('*')][i].append(br)
            arcs, well_formed = zip(*map(self._decode, planes))
            arcs = sorted(set(flatten(arcs)))
            if rels is not None:
                arcs = self.decode_rels(arcs, rels)
            return arcs, all(well_formed)
        
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

        def _decode(self, brackets: List[List[str]]) -> Tuple[List[Arc], bool]:
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
        input_tkzs: List[InputTokenizer], 
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        join_rels: bool,
        root_rel: Optional[str],
        k: int, 
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, join_rels, root_rel, device)
        self.k = k 
        self.lab = self.Labeler(k)
     
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = SemanticSLParser.add_arguments(argparser)
        argparser.add_argument('-k', type=int, help='Number of bracket planes')
        return argparser 
    
    @classmethod 
    def build(
        cls,
        data: Union[SDP, EnhancedCoNLL, str], 
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None, 
        char_conf: Optional[Config] = None,
        k: int = 2,
        join_rels: bool = False,
        device: int = 0,
        **_
    ) -> BracketSemanticParser:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        bracket_tkz, rel_tkz, rel_conf, root_rel = cls.build_targets(
            label_tkz=TargetTokenizer('BRACKET'), rel_tkz=TargetTokenizer('REL'), 
            labeler=cls.Labeler(k=k), data=data, join_rels=join_rels
        )
        return cls(input_tkzs, [bracket_tkz, rel_tkz], [enc_conf, *in_confs, bracket_tkz.conf, rel_conf], join_rels, root_rel, k, device)
