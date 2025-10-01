from separ.models.dep.idx.parser import IndexDependencyParser
from separ.models.dep.dit import DependencyDiTParser

class IndexDependencyDiTParser(IndexDependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-idx'
    
