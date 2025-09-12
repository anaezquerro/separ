from __future__ import annotations
from typing import Tuple, List, Optional, Union
import os 

from separ.models.grp.sl import GraphSLParser
from separ.data import TargetTokenizer, SDP, EnhancedCoNLL, Graph, Arc
from separ.utils import Config
from separ.models.dep.hier.parser import HierarchicalDependencyParser


class HierarchicalGraphParser(GraphSLParser):
    """Hierarchical Bracketing Semantic parser."""
    
    NAME = 'grp-hier'
    DECOLLAPSE = True 
    
    class Labeler(GraphSLParser.Labeler, HierarchicalDependencyParser.Labeler):
        
        def __init__(self):
            self.variant = 'nonp'
        
        def __repr__(self):
            return 'HierarchicalBracketSemanticLabeler()'
        
        def encode(self, graph: Graph) -> Tuple[List[str], List[str]]:
            graph = graph.collapse_one_cycles() # collapse cycles of length 1
            labels = self.encode_nonp(graph)
            return labels, self.encode_rels(graph.arcs, len(graph))

        def decode(self, labels: List[str], rels: Optional[List[str]] = None) -> Tuple[List[Arc], bool]:
            stack, arcs = [(0, HierarchicalDependencyParser.Bracket('/', structural=True, p=0, match=None))], []
            well_formed = True
            for i, label in enumerate(labels):
                brackets = HierarchicalDependencyParser.Bracket.from_string(label)
                for bracket in brackets:
                    added = False
                    if bracket.is_opening():
                        stack.append((i+1, bracket))
                        continue 
                    elif not bracket.structural: # closing auxiliar arc
                        # search for the p-th top opening structural arc 
                        for idx, s in stack[::-1]:
                            if s.structural and bracket.p == 0:
                                arcs.append(Arc(i+1, idx, None) if bracket.is_head() else Arc(idx, i+1, None))
                                added = True 
                                break 
                            elif s.structural:
                                bracket.p -= 1
                    else:
                        new = []
                        for j, (idx, s) in enumerate(stack[::-1]):
                            if s.structural and s.side == bracket.side and bracket.p == 0:
                                arcs.append(Arc(i+1, idx, None) if bracket.is_head() else Arc(idx, i+1, None))
                                added = True
                                break
                            elif s.structural and s.side == bracket.side:
                                bracket.p -= 1
                                new.append((idx, s))
                            elif s.structural:
                                new.append((idx, s))
                            else:
                                if s.p > 0:
                                    s.p -= 1
                                    new.append((idx, s))
                                else:
                                    arcs.append(Arc(idx, i+1, None) if s.is_head() else Arc(i+1, idx, None))
                                    added = True 
                        stack = stack[:-(j+1)] + new[::-1]
                    well_formed = well_formed and added 
            arcs = sorted(set(arcs))
            if rels is not None:
                arcs = self.decode_rels(arcs, rels)
            return arcs, len(stack) <= 1 and well_formed
                            
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lab = self.Labeler()
        
    @classmethod 
    def build(
        cls,
        data: Union[SDP, EnhancedCoNLL, str], 
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Optional[Config] = None, 
        char_conf: Optional[Config] = None,
        join_rels: bool = False,
        evalb: str | None = None,
        device: int = 0,
        **_
    ) -> HierarchicalGraphParser:
        if isinstance(data, str):
            data = cls.load_data(data)
        
        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf, tag_conf, char_conf)
        bracket_tkz, rel_tkz, rel_conf, root_rel = cls.build_targets(
            label_tkz=TargetTokenizer('BRACKET'), rel_tkz=TargetTokenizer('REL'), 
            labeler=cls.Labeler(), data=data, join_rels=join_rels
        )
        return cls(
            input_tkzs=input_tkzs,
            target_tkzs=[bracket_tkz, rel_tkz], 
            model_confs=[enc_conf, *in_confs, bracket_tkz.conf, rel_conf], 
            join_rels=join_rels, 
            root_rel=root_rel,
            evalb=evalb,
            device=device
        )

    