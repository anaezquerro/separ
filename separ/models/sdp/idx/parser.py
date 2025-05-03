from __future__ import annotations
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Union

from separ.data import SDP, InputTokenizer, TargetTokenizer, EnhancedCoNLL, Arc, Graph 
from separ.utils import Config
from separ.models.sdp.parser import SemanticSLParser


class IndexSemanticParser(SemanticSLParser):
    """
    Index Semantic Parser from [Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/).
    """
    NAME = 'sdp-idx'
    PARAMS = ['rel', 'join_rels', 'root_rel']
    
    class Labeler(SemanticSLParser.Labeler):
        
        def __init__(self, rel: bool):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexSemanticLabeler(rel={self.rel})'
            
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            indexes = [[] for _ in range(len(graph))]
            rels = [[] for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1].append(arc.HEAD - (arc.DEP*self.rel))
                rels[arc.DEP-1].append(arc.REL)
            indexes = ['' if len(index) == 0 else self.SEP.join(map(str, sorted(index))) for index in indexes]
            return indexes, [self.SEP.join(rel) for rel in rels]
        
        def decode(self, indexes: List[str], rels: Optional[List[str]] = None) -> Tuple[List[Arc], bool]:
            n, arcs, well_formed = len(indexes), [], True
            rels = [[] for _ in indexes] if rels is None else [rel.split(self.SEP) for rel in rels]
            for idep, index in enumerate(indexes):
                if index == '':
                    continue 
                for idx in index.split(self.SEP):
                    head = int(idx)+(idep+1)*self.rel 
                    if head in range(n+1):
                        arcs.append(Arc(head, idep+1, rels[idep].pop(0) if len(rels[idep]) > 0 else self.REL))
                    else:
                        well_formed = False
            return sorted(arcs), well_formed
        
    def __init__(
        self, 
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer], 
        model_confs: List[Config],
        join_rels: bool,
        root_rel: Optional[str],
        rel: bool,
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, join_rels, root_rel, device)
        self.rel = rel
        self.lab = self.Labeler(rel)
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = SemanticSLParser.add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    
    @classmethod 
    def build(
        cls, 
        data: Union[SDP, EnhancedCoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
        rel: bool = False,
        join_rels: bool = False,
        device: int = 0,
        **_
    ) -> IndexSemanticParser:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        index_tkz, rel_tkz, rel_conf, root_rel = cls.build_targets(
            label_tkz=TargetTokenizer('INDEX'), rel_tkz=TargetTokenizer('REL'), 
            labeler=cls.Labeler(rel=rel), data=data, join_rels=join_rels
        )
        return cls(input_tkzs, [index_tkz, rel_tkz], [enc_conf, *in_confs, index_tkz.conf, rel_conf], join_rels, root_rel, rel, device)
        