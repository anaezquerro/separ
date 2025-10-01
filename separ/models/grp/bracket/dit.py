from __future__ import annotations

from separ.data import SDP, EnhancedCoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config 
from separ.models.grp.bracket.parser import BracketGraphParser
from separ.models.grp.dit import GraphDiTParser

class BracketGraphDiTParser(BracketGraphParser, GraphDiTParser):
    NAME = 'grp-dit-bracket'
