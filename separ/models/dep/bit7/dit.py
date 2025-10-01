from __future__ import annotations

from separ.models.dep.bit7.parser import Bit7DependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar

class Bit7DependencyDiTParser(Bit7DependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-bit7'
    