from __future__ import annotations

from separ.data import SDP, EnhancedCoNLL, TargetTokenizer
from separ.utils import Config
from separ.models.grp.bit6k.parser import Bit6kGraphParser
from separ.models.grp.gat import GraphGaTParser

class Bit6kGraphGaTParser(Bit6kGraphParser, GraphGaTParser):
    NAME = 'grp-gat-bit6k'
    
