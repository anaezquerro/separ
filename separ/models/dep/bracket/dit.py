from __future__ import annotations

from separ.models.dep.bracket.parser import BracketDependencyParser
from separ.models.dep.dit import DependencyDiTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar

class BracketDependencyDiTParser(BracketDependencyParser, DependencyDiTParser):
    NAME = 'dep-dit-bracket'
    