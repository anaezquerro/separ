from .idx import *
from .pos import *
from .eager import *
from .bracket import *
from .bit4 import *
from .bit7 import *

import os 
from separ.data import CoNLL
from separ.utils import parallel

DEP_TEST = [
    IndexDependencyParser.Labeler(rel=False),
    IndexDependencyParser.Labeler(rel=True),
    PoSDependencyParser.Labeler(),
    ArcEagerDependencyParser.System(n_stack=1, n_buffer=1),
    BracketDependencyParser.Labeler(k=1),
    BracketDependencyParser.Labeler(k=2),
    Bit4DependencyParser.Labeler(),
    Bit7DependencyParser.Labeler(),
]

DEP_PARSERS = [IndexDependencyParser, PoSDependencyParser, BracketDependencyParser, Bit4DependencyParser, Bit7DependencyParser]

def test(data: CoNLL, num_workers: int = os.cpu_count()):
    for lab in DEP_TEST:
        try:
            assert all(parallel(lab.test, data, num_workers=num_workers))
        except:
            print(f'Error in {lab} parser')
            raise AssertionError
    
        