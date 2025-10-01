from __future__ import annotations

from separ.models.grp.bit6k.gat import Bit6kGraphGaTParser
from separ.models.grp.bit4k.parser import Bit4kGraphParser

class Bit4kGraphGaTParser(Bit6kGraphGaTParser, Bit4kGraphParser):
    NAME = 'grp-gat-bit4k'
    