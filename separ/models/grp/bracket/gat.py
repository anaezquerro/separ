from __future__ import annotations

from separ.data import SDP, EnhancedCoNLL, TargetTokenizer
from separ.utils import Config
from separ.models.grp.bracket.parser import BracketGraphParser
from separ.models.grp.gat import GraphGaTParser

class BracketGraphGaTParser(BracketGraphParser, GraphGaTParser):
    NAME = 'grp-gat-bracket'
