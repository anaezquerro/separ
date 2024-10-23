from .idx import *

import os 
from separ.data import PTB
from separ.utils import parallel

CON_TEST = [
    IndexConstituencyParser.Labeler(rel=False),
    IndexConstituencyParser.Labeler(rel=True)
]
CON_PARSERS = [IndexConstituencyParser]

def test(data: PTB, num_workers: int = os.cpu_count()):
    for lab in CON_TEST:
        if not all(parallel(lab.test, data, num_workers=num_workers)):
            print(f'Error in {lab} parser')