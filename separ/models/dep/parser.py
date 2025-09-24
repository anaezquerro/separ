from separ.data import CoNLL, InputTokenizer, PretrainedTokenizer, CharacterTokenizer
from separ.utils import DependencyMetric, Config
from separ.parser import Parser

class DependencyParser(Parser):
    DATASET = [CoNLL]
    
    @property
    def official(self) -> bool:
        return False 
    
    @property 
    def METRIC(self) -> DependencyMetric:
        return DependencyMetric()

    @classmethod
    def build_inputs(
        cls, 
        data: CoNLL,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
    ) -> tuple[list[InputTokenizer], list[Config]]:
        if hasattr(word_conf, 'pretrained'):
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)

            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
                
        return input_tkzs, in_confs