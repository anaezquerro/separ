from __future__ import annotations
import torch 

from separ.models.dep.hexa.parser import HexaTaggingDependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar

class HexaTaggingDependencyGaTParser(HexaTaggingDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-hexa'

    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.HEXA, tree.FENCE, tree.REL = self.lab.encode(tree)
            tree.HEXA = ['<pad>']*len(tree) if len(tree.HEXA) == 0 else ['<pad>'] + tree.HEXA + ['<pad>']
            tree.FENCE += ['<pad>']*(len(tree) - len(tree.FENCE))
            tree.transformed = True 
        return tree
    
    def collate(self, trees: list[CoNLL.Tree]):
        return DependencyGaTParser.collate(self, trees)
    
    def _pred(self, tree: CoNLL.Tree, hexas: torch.Tensor, fences: torch.Tensor, rels: torch.Tensor) -> tuple[CoNLL.Tree, bool]:
        return HexaTaggingDependencyParser._pred(self, tree, hexas[1:-1], fences[:-1], rels)
    
    @classmethod
    def build(
        cls,
        data: str | CoNLL,
        word_conf: Config,
        dis_conf: Config,
        proj: str | None = 'head',
        device: int = 0,
        **_ 
    ) -> HexaTaggingDependencyGaTParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        dis_conf |= word_tkz.conf
        
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
            model_confs=[word_conf, dis_conf, hexa_tkz.conf, fence_tkz.conf, rel_tkz.conf], 
            proj=proj,
            device=device 
        )