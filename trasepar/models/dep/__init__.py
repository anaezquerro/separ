from .idx import *
from .pos import *
from .eager import *
from .bracket import *
from .bit4 import *
from .bit7 import *
from .biaffine import *
from .hexa import *

DEP_PARSERS = [
    IndexDependencyParser,
    PoSDependencyParser,
    BracketDependencyParser,
    Bit4DependencyParser,
    Bit7DependencyParser,
    HexaTaggingDependencyParser,
    BiaffineDependencyParser,
]
