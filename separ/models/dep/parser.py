from argparse import ArgumentParser

from separ.data import CoNLL
from separ.utils import DependencyMetric 
from separ.parser import Parser

class DependencyParser(Parser):
    DATASET = [CoNLL]
    
    @property
    def official(self) -> bool:
        return False 
    
    @property 
    def METRIC(self) -> DependencyMetric:
        return DependencyMetric()
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser) -> ArgumentParser:
        return Parser.add_arguments(argparser)