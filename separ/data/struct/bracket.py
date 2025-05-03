from __future__ import annotations
from typing import List, Tuple


class Bracket:
    PRIORITY = ['\\', '>', '<', '/']
    PLANE = '*'
    
    def __init__(self, symbol: str, p: int = 0):
        self.symbol = symbol
        self.p = p
        
    @property
    def order(self) -> Tuple[int, int]:
        return self.PRIORITY.index(self.symbol), self.p
                    
    def __repr__(self) -> str:
        return f'{self.symbol}' + self.PLANE*self.p
    
    def __lt__(self, other: Bracket) -> str:
        return self.order < other.order
    
    def is_dep(self) -> bool:
        return self.symbol == '>' or self.symbol == '<'
    
    def is_head(self) -> bool:
        return self.symbol == '\\' or self.symbol == '/'
    
    def is_closing(self) -> bool:
        return self.symbol == '>' or self.symbol == '\\'
    
    def is_opening(self) -> bool:
        return self.symbol == '<' or self.symbol == '/'
    
    def is_left(self) -> bool:
        return self.symbol  == '\\' or self.symbol == '<'
    
    def is_right(self) -> bool:
        return self.symbol == '>' or self.symbol == '/'
    
    @property
    def side(self) -> int:
        if self.is_left():
            return -1 
        else:
            return 1
        
    @classmethod
    def from_string(cls, raw: str) -> List[Bracket]:
        brackets = []
        for c in raw:
            if c in cls.PRIORITY:
                brackets.append(c)
            else:
                brackets[-1] += c 
        return [cls(bracket[0], bracket.count(cls.PLANE)) for bracket in brackets]