from __future__ import annotations
from typing import List, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader
import random, tempfile, os

from separ.utils.sampler import TokenizedSampler, StrictTokenizationSampler
        
    
class AbstractDataset(Dataset):
    SEP = None
    EXTENSION = None
    END = ''
    HEADER = ''
    
    def __init__(self, sents: list, path: str):
        self.sents = sents 
        self.path = path
        for i, sen in enumerate(self.sents):
            if sen.ID is None:
                sen.ID = i 
        self.sort()
        
    @property
    def name(self) -> str:
        return self.path.split('/')[-1].split('.')[0]
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(path={self.path}, n={len(self)})'
                
    def sort(self) -> AbstractDataset:
        self.sents = sorted(self.sents, key=lambda sen: sen.ID)
        return self 
        
    def __iter__(self):
        return iter(self.sents)
    
    def __getitem__(self, index: Union[int, slice, list, tuple]):
        if isinstance(index, int):
            return self.sents[index]
        elif isinstance(index, slice):
            return self.__class__(self.sents[index], self.path)
        elif isinstance(index, list) or isinstance(index, tuple):
            return self.__class__([self.sents[i] for i in index], self.path)
    
    def __len__(self) -> int:
        return len(self.sents)
    
    def split(self, p: float, shuffle: bool = True) -> Tuple[AbstractDataset, AbstractDataset]:
        if shuffle:
            random.shuffle(self.sents)
        n = int(p*len(self))
        split = self.sents[:n]
        self.sents = self.sents[n:]
        return self, self.__class__(split, self.path)
        
    def copy(self) -> AbstractDataset:
        return self.__class__([sen.copy() for sen in self.sents], self.path)
    
    def save(self, path: str):
        self.sort()
        with open(path, 'w') as writer:
            if len(self.HEADER) > 0:
                writer.write(self.HEADER + '\n')
            writer.write(self.SEP.join(sen.format() for sen in self.sents))
            writer.write(self.END)
            
    def join(self, *others) -> AbstractDataset:
        sents = self.sents 
        for other in others:
            sents += other.sents
        return self.__class__(sents, self.path)
            
    @classmethod
    def from_files(cls, paths: List[str], num_workers: int = 1) -> AbstractDataset:
        datas = [cls.from_file(path, num_workers=num_workers) for path in paths]
        data = datas.pop(0)
        data.join(*datas)
        return data

    @property
    def n_tokens(self) -> int:
        return sum(map(len, self.sents))
    
    @property
    def lens(self) -> List[int]:
        return list(map(len, self.sents))
    
    def sampler(self, batch_size: int, pred: bool = False) -> Union[StrictTokenizationSampler, TokenizedSampler]:
        if pred:
            return StrictTokenizationSampler(self, batch_size=batch_size, shuffle=False)
        else:
            return TokenizedSampler(self, batch_size=batch_size, shuffle=True)
        
    def loader(self, collate: Callable, batch_size: int, pred: bool = False) -> DataLoader:
        return DataLoader(self, batch_sampler=self.sampler(batch_size, pred), collate_fn=collate)
    
    @classmethod
    def test(cls, path: str) -> bool:
        """Checks if the dataset representation does not lose information (load -> save produces 
        the same raw text).

        Args:
            path (str): Path to load the dataset.

        Returns:
            bool: Whether the representation does not lose information.
        """
        data = cls.from_file(path)
        filename = tempfile.NamedTemporaryFile(mode="w", suffix=cls.EXTENSION).name 
        data.save(filename)
        result = open(filename, 'r').read() == open(path, 'r').read()
        os.remove(filename)
        return result 