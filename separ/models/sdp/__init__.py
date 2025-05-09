from .idx import *
from .bracket import BracketSemanticParser
from .bit4k import Bit4kSemanticParser
from .bit6k import Bit6kSemanticParser
from .biaffine import BiaffineSemanticParser
from .cov import CovingtonSemanticParser

SDP_PARSERS = [
    IndexSemanticParser, BracketSemanticParser, Bit4kSemanticParser, Bit6kSemanticParser, 
    BiaffineSemanticParser, CovingtonSemanticParser
]