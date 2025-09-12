from __future__ import annotations

from separ.models.dep.hexa.parser import HexaTaggingDependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import TargetTokenizer, PretrainedTokenizer, CoNLL, Arc
from separ.utils import Config, flatten, bar

class HexaTaggingDependencyDiTParser(HexaTaggingDependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-hexa'
    
    class Labeler(HexaTaggingDependencyParser.Labeler):
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str], list[str]]:
            if len(tree) == 1:
                return ['>'], ['<pad>'], tree.DEPREL
            else:
                hexas, fences, rels = super().encode(tree)
                return ['>'] + hexas + ['<'], ['<pad>'] + fences, rels
        
        def decode(self, hexas: list[str], fences: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            return super().decode(hexas[1:-1], fences[1:], rels)
    
    def collate(self, trees: list[CoNLL.Tree]):
        return DependencyDiTParser.collate(self, trees)
    
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        dec_conf: Config,
        proj: str | None = None,
        device: int = 0,
        **_
    ) -> HexaTaggingDependencyDiTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        
        hexa_tkz = TargetTokenizer('HEXA')
        fence_tkz = TargetTokenizer('FENCE')
        rel_tkz = TargetTokenizer('REL')
        labeler = cls.Labeler(proj)
        hexas, fences, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        hexa_tkz.train(hexas)
        fence_tkz.train(fences)
        rel_tkz.train(rels)
        
        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=[hexa_tkz, fence_tkz, rel_tkz],
            model_confs=[enc_conf, word_conf, dec_conf, hexa_tkz.conf, fence_tkz.conf, rel_tkz.conf],
            proj=proj, 
            device=device 
        )