from __future__ import annotations

from separ.models.grp.bit4k.parser import Bit4kGraphParser
from separ.models.grp.bit6k.dit import Bit6kGraphDiTParser

class Bit4kGraphDiTParser(Bit6kGraphDiTParser, Bit4kGraphParser):
    NAME = 'grp-dit-bit4k'
    