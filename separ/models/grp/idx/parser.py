from __future__ import annotations
from argparse import ArgumentParser

from separ.data import SDP, TargetTokenizer, EnhancedCoNLL, Arc, Graph 
from separ.utils import Config
from separ.models.grp.sl import GraphSLParser


class IndexGraphParser(GraphSLParser):
    """
    Index Graph Parser from [Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/).
    """
    NAME = 'grp-idx'
    PARAMS = GraphSLParser.PARAMS | {'rel'}
    
    class Labeler(GraphSLParser.Labeler):
        
        def __init__(self, rel: bool):
            self.rel = rel 
            
        def __repr__(self) -> str:
            return f'IndexSemanticLabeler(rel={self.rel})'
            
        def encode(self, graph: Graph) -> tuple[list[str], list[str]]:
            indexes = [[] for _ in range(len(graph))]
            rels = [[] for _ in range(len(graph))]
            for arc in graph.arcs:
                indexes[arc.DEP-1].append(arc.HEAD - (arc.DEP*self.rel))
                rels[arc.DEP-1].append(arc.REL)
            indexes = ['' if len(index) == 0 else self.SEP.join(map(str, sorted(index))) for index in indexes]
            return indexes, [self.SEP.join(rel) for rel in rels]
        
        def decode(self, indexes: list[str], rels: list[str] | None = None) -> tuple[list[Arc], bool]:
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
        
    def __init__(self, rel: bool, **kwargs):
        super().__init__(**kwargs)
        self.rel = rel
        self.lab = self.Labeler(rel)
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super().add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser 
    
    @classmethod 
    def build(
        cls, 
        data: str | SDP | EnhancedCoNLL,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        rel: bool = False,
        join_rels: bool = False,
        evalb: str | None = None,
        device: int = 0,
        **_
    ) -> IndexGraphParser:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        index_tkz, rel_tkz, rel_conf, root_rel = cls.build_targets(
            label_tkz=TargetTokenizer('INDEX'), rel_tkz=TargetTokenizer('REL'), 
            labeler=cls.Labeler(rel=rel), data=data, join_rels=join_rels
        )
        return cls(
            input_tkzs=input_tkzs, 
            target_tkzs=[index_tkz, rel_tkz], 
            model_confs=[enc_conf, *in_confs, index_tkz.conf, rel_conf], 
            rel=rel,
            join_rels=join_rels, 
            root_rel=root_rel,
            evalb=evalb,
            device=device
        )
        