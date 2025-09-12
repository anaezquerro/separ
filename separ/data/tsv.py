from __future__ import annotations
import re 

from separ.data.struct import Dataset, Sentence
from separ.utils.logger import bar 

class TSV(Dataset):
    """Abstract representation of a TSV dataset with arbitrary fields."""
    SEP = '\n\n'
    EXTENSION = 'tsv'
    
    class Sentence(Sentence):
        
        class Node(Sentence.Node):
            def __init__(self, FIELDS: list[str], *args):
                self.FIELDS = FIELDS 
                assert len(FIELDS) == len(args), 'Missing fields'
                super().__init__(*args)
                
            def copy(self) -> Sentence.Node:
                return self.__class__(self.FIELDS, *map(self.__dict__.get, self.FIELDS))
                
            @classmethod
            def from_raw(cls, line: str, FIELDS: list[str]):
                return cls(FIELDS, *re.split(r'[ \t]+', line.strip()))
        
        def __init__(
            self,
            nodes: list[Sentence.Node], 
            FIELDS: list[str],
            ID: int | None = None,
            annotations: list[str | int] | None = None
        ):
            self._FIELDS = FIELDS 
            super().__init__(nodes, ID, annotations)
            
        @property
        def FIELDS(self) -> list[str]:
            return self._FIELDS
            
        @classmethod
        def from_raw(cls, block: str, ID: int, FIELDS: list[str]) -> TSV.Sentence:
            lines, annotations = [], []
            for line in block.strip().split(cls.SEP):
                if not line.startswith('#'):
                    annotations.append(len(lines))
                    lines.append(line)
                else:
                    annotations.append(line)
            nodes = [cls.Node.from_raw(line, FIELDS) for line in lines]
            return cls(nodes, FIELDS, ID, annotations)
        
    @classmethod
    def from_file(cls, path: str) -> TSV:
        raw = open(path, 'r').read().strip()
        assert raw.startswith('#'), 'The first line of the file must include the fields'
        FIELDS = raw.split('\n')[0][1:].strip().split()
        blocks = re.split('\n{2,}', raw)
        sens = [cls.Sentence.from_raw(block, ID, FIELDS) for ID, block in 
                bar(enumerate(blocks), total=len(blocks), leave=False, desc=path)]
        return cls(sens, path)