from __future__ import annotations

from separ.utils.fn import flatten 

class Sentence:
    SEP: str = '\n'
    class Node:
        SEP: str  = '\t'
        FIELDS: list[str]

        def __init__(self, *args):
            for field, arg in zip(self.FIELDS, args):
                self.__setattr__(field, arg)
        
        def format(self) -> str:
            return self.SEP.join(map(str, flatten(self.__getattribute__(field) for field in self.FIELDS)))
        
        def copy(self) -> Sentence.Node:
            return self.__class__(*map(self.__dict__.get, self.FIELDS))
        
        def __eq__(self, other: Sentence.Node) -> bool:
            if isinstance(other, Sentence.Node):
                return len(set(self.FIELDS) - set(other.FIELDS)) == 0 \
                    and all(getattr(self, field) == getattr(other, field) for field in self.FIELDS)
                    
        def __lt__(self, other: Sentence.Node) -> bool:
            return self.ID < other.ID 
        
        @classmethod
        def from_raw(cls, line: str) -> Sentence.Node:
            return cls(*line.strip().split(cls.SEP))
        
    def __init__(
        self, 
        nodes: list[Sentence.Node], 
        ID: int | None = None, 
        annotations: list[int | str] | None = None
    ):
        self.nodes = nodes 
        self.ID = ID 
        if annotations is not None:
            self.annotations = annotations
        else:
            self.annotations = list(range(len(nodes)))
            
    def copy(self) -> Sentence:
        return Sentence(nodes=[node.copy() for node in self.nodes], ID=self.ID, annotations=self.annotations)
    
    @property
    def FIELDS(self) -> list[str]:
        return self.Node.FIELDS[1:]
        
    def format(self) -> str:
        return self.SEP.join(self.nodes[x].format() if isinstance(x, int) else x for x in self.annotations)
        
    def __getattr__(self, name: str) -> list[str]:
        """Gets the values of an specific field for each token."""
        if name in self.FIELDS:
            return [getattr(node, name) for node in self.nodes]
        else:
            return object.__getattribute__(self, name)
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self) -> list[Sentence.Node]:
        return iter(self.nodes)
    
    def rebuild(self, field: str, values: list[str]) -> Sentence:
        """Rebuilds the sentence with a new field.

        Args:
            field (str): Name of the field to rebuild.
            values (list[str]): New values of the field.

        Returns:
            Sentence.
        """
        new = self.copy()
        for node, value in zip(new.nodes, values):
            node.__setattr__(field, value)
        return new 