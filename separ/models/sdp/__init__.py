from .idx import *
from .bracket import *
from .bit4k import *
from .bit6k import *


from separ.data import SDP
from separ.utils import parallel

SDP_TEST = [
    IndexSemanticParser.Labeler(rel=False),
    IndexSemanticParser.Labeler(rel=True),
    BracketSemanticParser.Labeler(k=2),
    BracketSemanticParser.Labeler(k=3),
    Bit4kSemanticParser.Labeler(k=2),
    Bit4kSemanticParser.Labeler(k=3),
    Bit4kSemanticParser.Labeler(k=4),
    Bit6kSemanticParser.Labeler(k=2),
    Bit6kSemanticParser.Labeler(k=3),
    Bit6kSemanticParser.Labeler(k=4),
]

SDP_PARSERS = [IndexSemanticParser, BracketSemanticParser, Bit4kSemanticParser,Bit6kSemanticParser]

def test(data: SDP, num_workers: int = 1):
    for lab in SDP_TEST:
        if not all(parallel(lab.test, data, num_workers=num_workers)):
            print(f'Error in {lab} parser')
    
        