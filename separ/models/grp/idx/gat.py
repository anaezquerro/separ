from __future__ import annotations 

from separ.models.grp.idx.parser import IndexGraphParser
from separ.models.grp.gat import GraphGaTParser

class IndexGraphGaTParser(IndexGraphParser, GraphGaTParser):
    NAME = 'grp-gat-idx'
    
