from __future__ import annotations

from separ.models.dep.bit4.parser import Bit4DependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar 

class Bit4DependencyGaTParser(Bit4DependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-bit4'
    
    @classmethod
    def build(
        cls,
        data: str | CoNLL,
        word_conf: Config,
        dis_conf: Config,
        proj: str | None = None,
        device: int = 0,
        **_ 
    ) -> Bit4DependencyGaTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        dis_conf |= word_tkz.conf
        
        bit_tkz, rel_tkz = TargetTokenizer('BIT'), TargetTokenizer('REL')
        labeler = cls.Labeler(proj)
        bits, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bit_tkz.train(bits)
        rel_tkz.train(rels)
        
        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=[bit_tkz, rel_tkz], 
            model_confs=[word_conf, dis_conf, bit_tkz.conf, rel_tkz.conf], 
            proj=proj,
            device=device
        )