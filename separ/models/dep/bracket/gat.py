from __future__ import annotations

from separ.models.dep.bracket.parser import BracketDependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar


class BracketDependencyGaTParser(BracketDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-bracket'
    
    @classmethod
    def build(
        cls,
        data: str | CoNLL,
        word_conf: Config,
        dis_conf: Config,
        k: int = 1,
        device: int = 0,
        **_ 
    ) -> BracketDependencyGaTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        dis_conf |= word_tkz.conf
        
        bracket_tkz, rel_tkz = TargetTokenizer('BRACKET'), TargetTokenizer('REL')
        labeler = cls.Labeler(k)
        brackets, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bracket_tkz.train(brackets)
        rel_tkz.train(rels)
        
        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=[bracket_tkz, rel_tkz], 
            model_confs=[word_conf, dis_conf, bracket_tkz.conf, rel_tkz.conf],
            k=k,
            device=device
        )
    