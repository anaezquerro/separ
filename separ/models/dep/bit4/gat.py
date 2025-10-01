from __future__ import annotations

from separ.models.dep.bit4.parser import Bit4DependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar 

class Bit4DependencyGaTParser(Bit4DependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-bit4'
    