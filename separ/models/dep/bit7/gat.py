from __future__ import annotations

from separ.models.dep.bit7.parser import Bit7DependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.utils import Config, flatten, bar
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer


class Bit7DependencyGaTParser(Bit7DependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-bit7'
