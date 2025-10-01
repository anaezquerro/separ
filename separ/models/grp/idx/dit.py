from __future__ import annotations

from separ.models.grp.dit import GraphDiTParser
from separ.models.grp.idx.parser import IndexGraphParser


class IndexGraphDiTParser(IndexGraphParser, GraphDiTParser):
    NAME = 'grp-dit-idx'
    

