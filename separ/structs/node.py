from __future__ import annotations
from typing import List, Union
from separ.utils.fn import flatten 

class AbstractNode:
    """Abstract representation of a node in a graph."""
    SEP = '\t'
    BOS = '<bos>'
    EOS = '<eos>'
    FIELDS = ['ID']  # mandatory field 
    
    def __init__(self, ID: int):
        self.ID = ID
    
    def __repr__(self) -> str:
        return self.format()
    
    def __eq__(self, other: AbstractNode) -> bool:
        return isinstance(other, self.__class__) and \
            all(v1 == v2 for v1, v2 in zip(self.values(), other.values()))
    
    def __le__(self, other: AbstractNode) -> bool:
        return self.ID <= other.ID 
    
    def __lt__(self, other: AbstractNode) -> bool:
        return self.ID < other.ID 
    
    def __ge__(self, other: AbstractNode) -> bool:
        return self.ID >= other.ID 
    
    def __gt__(self, other: AbstractNode) -> bool:
        return self.ID > other.ID
    
    def is_bos(self) -> bool:
        return self.ID == 0 
    
    def values(self) -> List[Union[str, int]]:
        return flatten(self.__getattribute__(field) for field in self.FIELDS)
    
    def format(self) -> str:
        assert not any(value is None for value in self.values()),'NULL values cannot be formatted'
        return self.SEP.join(map(str, self.values()))
    
    def copy(self) -> AbstractNode:
        return self.__class__(*self.values())
    
    @classmethod
    def bos(cls) -> AbstractNode:
        """Create a BoS node with ID set to 0."""
        return cls(0, *[cls.BOS if field != 'HEAD' else None for field in cls.FIELDS[1:]])
    
    @classmethod
    def eos(cls, n: int) -> AbstractNode:
        """Creates an EoS node with ID set to `n`."""
        return cls(n, *[cls.EOS if field != 'HEAD' else None for field in cls.FIELDS[1:]])
    

    