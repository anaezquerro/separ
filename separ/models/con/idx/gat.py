from __future__ import annotations

from separ.models.con.idx.parser import IndexConstituencyParser
from separ.models.con.gat import ConstituencyGaTParser
from separ.utils import Config, flatten, bar
from separ.data import PTB, PretrainedTokenizer, TargetTokenizer

class IndexConstituencyGaTParser(IndexConstituencyParser, ConstituencyGaTParser):
    NAME = 'con-gat-idx'

