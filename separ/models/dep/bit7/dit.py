from __future__ import annotations

from separ.models.dep.bit7.parser import Bit7DependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar

class Bit7DependencyDiTParser(Bit7DependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-bit7'
    
    @classmethod
    def build(
        cls,
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        dec_conf: Config,
        device: int,
        **_ 
    ) -> Bit7DependencyDiTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        
        bit_tkz, rel_tkz = TargetTokenizer('BIT'), TargetTokenizer('REL')
        labeler = cls.Labeler()
        bits, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        bit_tkz.train(bits)
        rel_tkz.train(rels)
        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=[bit_tkz, rel_tkz], 
            model_confs=[enc_conf, word_conf, dec_conf, bit_tkz.conf, rel_tkz.conf], 
            device=device
        )