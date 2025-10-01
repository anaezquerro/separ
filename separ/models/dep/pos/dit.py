from separ.models.dep.pos.parser import PoSDependencyParser
from separ.models.dep.dit import DependencyDiTParser

class PoSDependencyDiTParser(PoSDependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-pos'
    