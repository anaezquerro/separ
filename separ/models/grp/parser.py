from __future__ import annotations
from argparse import ArgumentParser

from separ.parser import Parser
from separ.data import SDP, EnhancedCoNLL
from separ.utils import GraphMetric

class GraphParser(Parser):
    DATASET = [SDP, EnhancedCoNLL]
    PARAMS = {'root_rel', 'evalb'}
    
    def __init__(self, root_rel: str | None, evalb: str | None, **kwargs):
        super().__init__(**kwargs)
        self.root_rel = root_rel 
        self.evalb = evalb
    
    @property
    def official(self) -> bool:
        if not hasattr(self, 'evalb'):
            self.evalb = None
        return self.evalb is not None 
        
    @property
    def METRIC(self) -> GraphMetric:
        if self.official:
            return GraphMetric(self.evalb)
        else:
            return GraphMetric()

    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(GraphParser, cls).add_arguments(argparser)
        argparser.add_argument('--evalb', default=None, type=str, required=False,  help='Evaluation script')
        return argparser