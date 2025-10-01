from separ.models.dep.idx.parser import IndexDependencyParser
from separ.models.dep.gat import DependencyGaTParser

class IndexDependencyGaTParser(IndexDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-idx'
    