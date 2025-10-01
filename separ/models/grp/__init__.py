from .idx import *
from .bracket import *
from .bit4k import *
from .bit6k import *
from .biaffine import *
from .cov import *
from .hier import *

SDP_PARSERS = [
    IndexGraphParser, BracketGraphParser, Bit4kGraphParser, Bit6kGraphParser, HierarchicalGraphParser, BiaffineGraphParser, CovingtonGraphParser,
    IndexGraphDiTParser, BracketGraphDiTParser, Bit4kGraphDiTParser, Bit6kGraphDiTParser,
    IndexGraphGaTParser, BracketGraphGaTParser, Bit4kGraphGaTParser, Bit6kGraphGaTParser
]