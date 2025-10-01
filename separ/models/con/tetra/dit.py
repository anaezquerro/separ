from __future__ import annotations

from separ.models.con.tetra.parser import TetraTaggingConstituencyParser
from separ.models.con.dit import ConstituencyDiTParser
from separ.data import PTB, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar 


class TetraTaggingConstituencyDiTParser(TetraTaggingConstituencyParser, ConstituencyDiTParser):
    NAME = 'con-dit-tetra'
    
    class Labeler(TetraTaggingConstituencyParser.Labeler):
        
        def encode(self, tree: PTB.Tree) -> tuple[list[str], list[str], list[str], list[str]]:
            tetras, fences, cons, *tags = super().encode(tree)
            if len(tree) > 1:
                return ['>'] + tetras + ['<'], ['>'] + fences + ['<'], cons + ['<pad>'], *tags
            else:
                return ['<pad>'], ['<pad>'], cons, *tags
        
        def decode(self, tetras: list[str], fences: list[str], cons: list[str], leaves: list[str], words: list[str]):
            return super().decode(tetras[1:-1], fences[1:-1], cons, leaves, words)

    def collate(self, trees: list[PTB.Tree]):
        return ConstituencyDiTParser.collate(self, trees)
