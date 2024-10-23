from __future__ import annotations
import plotly.graph_objects as go 
import plotly.express as px 
from typing import Dict, Optional, Iterable, List 
import pickle 
import numpy as np

from separ.utils.metric import Metric
from separ.utils.fn import flatten


class Epoch:
    def __init__(self, epoch: int):
        self.epoch = epoch 
        
    def add_subset(self, subset: str, **kwargs):
        self.__setattr__(subset, kwargs)
        
    def items(self) -> dict:
        return dict(epoch=self.epoch, **self.__dict__)
        

class History:
    def __init__(self):
        self.history = []
        self.ibest = 0
        
    def __len__(self) -> int:
        return len(self.history)
    
    def add(self, epoch: Epoch, best: bool):
        self.history.append(epoch)
        if best:
            self.ibest = epoch.epoch-1
        
    @property
    def best(self) -> Epoch:
        return self.history[self.ibest]

    def __iter__(self) -> Iterable[Epoch]:
        return iter(self.history)

    def save(self, path: str): 
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)
            
    @classmethod
    def load(cls, path: str) -> History:
        with open(path, 'rb') as reader:
            hist = pickle.load(reader)
        return hist 
        
                
        
        
        