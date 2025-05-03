from __future__ import annotations
import numpy as np 

class Arc:
    def __init__(self, HEAD: int, DEP: int, REL: str):
        assert DEP is None or DEP > 0
        if not isinstance(HEAD, bool) and not isinstance(DEP, bool):
            assert HEAD != DEP, 'Arc cannot be a cycle of length 1'
        self.HEAD = HEAD 
        self.DEP = DEP
        self.REL = REL 
        
    def __len__(self) -> int:
        return abs(self.HEAD - self.DEP)
    
    def __repr__(self) -> str:
        return f'{self.HEAD} --({self.REL})--> {self.DEP}'
    
    def __contains__(self, other: Arc):
        if isinstance(other, Arc):
            return other.HEAD in self.range and other.DEP in self.range
        else:
            raise NotImplementedError
    
    @property
    def side(self) -> int:
        return np.sign(self.DEP-self.HEAD)
    
    @property
    def left(self) -> int:
        return min(self.HEAD, self.DEP)
    
    @property
    def right(self) -> int:
        return max(self.HEAD, self.DEP)
        
    @property 
    def range(self) -> range:
        """Range of the arc in the graph, defined as the range of position between the 
        predicate and argument, included."""
        return range(self.left, self.right+1)
    
    @property
    def coverage(self) -> range:
        """Coverage of the arc i nthe graph, defined as the range of inner positions between 
        the predicate and argument, not included."""
        return range(self.left+1, self.right)
    
    def cross(self, other: Arc) -> bool:
        """Checks if it crosses with other arc."""
        if (other.left in self.coverage and other.right not in self.range) or \
            (other.right in self.coverage and other.left not in self.range) or \
                (self.right in other.coverage and self.left not in other.range) or \
                    (self.left in other.coverage and self.right not in other.range):
                        return True 
        else:
            return False
        
    def __eq__(self, other: Arc) -> bool:
        return (self.HEAD == other.HEAD) and (self.DEP == other.DEP) and (self.REL == other.REL)
    
    def __le__(self, other: Arc) -> bool:
        if self.DEP == other.DEP:
            return self.HEAD <= other.HEAD 
        else:
            return self.DEP < other.DEP 
        
    def __lt__(self, other: Arc) -> bool:
        if self.DEP == other.DEP:
            return self.HEAD < other.HEAD 
        else:
            return self.DEP < other.DEP 
        
    def __gt__(self, other: Arc) -> bool:
        if self.DEP == other.DEP:
            return self.HEAD > other.HEAD 
        else:
            return self.DEP > other.DEP 
        
    def __ge__(self, other: Arc) -> bool:
        if self.DEP == other.DEP:
            return self.HEAD >= other.HEAD 
        else:
            return self.DEP > other.DEP 
    
    def equals(self, other: Arc, labeled: bool = True) -> bool:
        if labeled:
            return self == other 
        else:
            return (self.HEAD == other.HEAD) and (self.DEP == other.DEP)
    
    def copy(self) -> Arc:
        return Arc(self.HEAD, self.DEP, self.REL)
    
    def is_labeled(self) -> bool:
        return self.REL is not None 
    
    def is_closed(self) -> bool:
        return isinstance(self.DEP, int) and isinstance(self.HEAD, int)
    
    def __hash__(self) -> int:
        return hash((self.HEAD, self.DEP, self.REL))
