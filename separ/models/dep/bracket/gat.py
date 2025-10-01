from __future__ import annotations

from separ.models.dep.bracket.parser import BracketDependencyParser
from separ.models.dep.gat import DependencyGaTParser
from separ.data import CoNLL, PretrainedTokenizer, TargetTokenizer
from separ.utils import Config, flatten, bar


class BracketDependencyGaTParser(BracketDependencyParser, DependencyGaTParser):
    NAME = 'dep-gat-bracket'
