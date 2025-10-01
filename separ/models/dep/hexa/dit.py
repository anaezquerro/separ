from __future__ import annotations

from separ.models.dep.hexa.parser import HexaTaggingDependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import CoNLL, Arc

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
    

    