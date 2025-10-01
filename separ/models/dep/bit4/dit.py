
from __future__ import annotations

from separ.models.dep.bit4.parser import Bit4DependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar

class Bit4DependencyDiTParser(Bit4DependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-bit4'
    