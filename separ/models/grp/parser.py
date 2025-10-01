from __future__ import annotations
from argparse import ArgumentParser

from separ.parser import Parser
from separ.data import SDP, EnhancedCoNLL, PretrainedTokenizer, InputTokenizer, CharacterTokenizer
from separ.utils import GraphMetric, Config

class GraphParser(Parser):
    DATASET = [SDP, EnhancedCoNLL]
    PARAMS = {'root_rel', 'evalb'}
    
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
        argparser.add_argument('--evalb', default=None, type=str, required=False, help='Evaluation script')
        return argparser
    
