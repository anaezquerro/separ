from .idx import *
from .tetra import *


CON_PARSERS = [
    IndexConstituencyParser, TetraTaggingConstituencyParser, 
    IndexConstituencyDiTParser, TetraTaggingConstituencyDiTParser,
    IndexConstituencyGaTParser, TetraTaggingConstituencyGaTParser
]