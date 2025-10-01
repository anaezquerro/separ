from __future__ import annotations

from separ.data import SDP, EnhancedCoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config 
from separ.models.grp.bit6k.parser import Bit6kGraphParser
from separ.models.grp.dit import GraphDiTParser

class Bit6kGraphDiTParser(Bit6kGraphParser, GraphDiTParser):
    NAME = 'grp-dit-bit6k'
    
