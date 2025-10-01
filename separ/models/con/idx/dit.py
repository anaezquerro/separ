from __future__ import annotations

from separ.models.con.idx.parser import IndexConstituencyParser
from separ.models.con.dit import ConstituencyDiTParser

class IndexConstituencyDiTParser(IndexConstituencyParser, ConstituencyDiTParser):
    NAME = 'con-dit-idx'
    
