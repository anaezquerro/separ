from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Callable
from torch.utils.data import DataLoader, Sampler
import torch, random 
import numpy as np 

from separ.data.sampler import TokenizedBatchSampler
from separ.utils.fn import filename, listdir
from separ.utils.common import WORLD_SIZE
from separ.data.struct.sentence import Sentence
    
class Dataset(torch.utils.data.Dataset):
    SEP: str
    EXTENSION: str
    
    def __init__(self, sens: List[Sentence], path: str):
        super().__init__()
        self.sens = sens 
        self.path = path 
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}(path={self.path}, n={len(self)})'
        
    @property
    def name(self) -> str:
        return filename(self.path, extension=False)
        
    def __len__(self) -> int:
        return len(self.sens)
    
    def sort(self) -> Dataset:
        self.sens = sorted(self.sens, key=lambda sen: sen.ID)
        return self
    
    def format(self) -> str:
        self.sort()
        return self.SEP.join(sen.format() for sen in self)
    
    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, slice):
            return self.sens[index]
        else:
            return list(map(self.sens.__getitem__, index))
        
    @property 
    def lens(self) -> Dict[int, int]:
        return {i: len(sen) for i, sen in enumerate(self.sens)}
    
    @property
    def n_tokens(self) -> int:
        return sum(map(len, self.sens))
    
    def loader(
        self, 
        batch_size: int, 
        shuffle: bool, 
        collate: Callable, 
        device: Optional[int] = None,
        lens: Optional[Dict[int, int]] = None
    ) -> Tuple[DataLoader, Sampler]:
        sampler = TokenizedBatchSampler(lens if lens is not None else self.lens, batch_size=batch_size, shuffle=shuffle, device=device)
        loader = DataLoader(self, batch_sampler=sampler, collate_fn=collate, shuffle=False, pin_memory=True)
        return loader, sampler 
    
    def split(self, device: int, shuffle: bool = False, seed: int = 0) -> Dataset:
        indices = list(range(len(self)))
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)
        indices = indices[device:len(indices):WORLD_SIZE]
        return self.__class__(sens=[self[i] for i in indices], path=self.path)
    
    def save(self, path: str):
        with open(path, 'w') as writer:
            writer.write(self.format())
            
    @classmethod
    def from_files(cls, paths: List[str]) -> Dataset:
        sens = []
        for path in paths:
            sens += cls.from_file(path).sens 
        return cls(sens, path=None)
    
    @classmethod
    def from_folder(cls, folder: str) -> Dataset:
        paths = [path for path in listdir(folder, absolute=True) if path.endswith(cls.EXTENSION)]
        return cls.from_files(paths)
    