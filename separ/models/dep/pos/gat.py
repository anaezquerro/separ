from separ.models.dep.gat import DependencyGaTParser
from separ.models.dep.pos.parser import PoSDependencyParser

class PoSDependencyGaTParser(PoSDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-pos'
    