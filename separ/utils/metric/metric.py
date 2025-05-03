from __future__ import annotations
from typing import List, Dict
import pickle, torch, os 
import torch.distributed as dist

from separ.utils.fn import mkftemp, folderpath 

class Metric:
    METRICS = [] 
    ATTRIBUTES = []
    KEY_METRICS = []
    
    def __init__(self, *args, **kwargs):
        """
        Abstract metric class. The metric stores values to different metrics and has synchronization capabilities.
        
        - ATTRIBUTES: Stores some indicators (such as #samples, #true-positives) to compute metrics.
        - METRICS: Evaluation metrics.
        - KEY_METRICS: Which metrics of METRICS are evaluated to compare different instances. 
        
        By default, those METRICS in uppercase would be displayed in % (thus the result would be scaled by 100).
        """
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, torch.tensor(1e-12))
        self.__setattr__('n', torch.tensor(0))
        
        if len(args) > 0 or len(kwargs) > 0:
            self(*args, **kwargs)
    
    def __eq__(self, other: Metric) -> bool:
        return isinstance(other, self.__class__) and all(getattr(self, attr) == getattr(other, attr) for attr in self.ATTRIBUTES)
    
    def __add__(self, other: Metric) -> Metric:
        assert self.__class__.__name__ == self.__class__.__name__,\
            f'Metrics must be the same: {self.__class__.__name__} != {other.__class__.__name__}'
        new = self.copy()
        for attr in new.ATTRIBUTES:
            new.__setattr__(attr, getattr(new, attr) + getattr(other, attr))
        return new
    
    def __radd__(self, other: Metric) -> Metric:
        return self + other if isinstance(other, Metric) else self 
    
    def __call__(self, *args, **kwargs) -> Metric:
        raise NotImplementedError
    
    def __repr__(self):
        return f', '.join(f'{name}={getattr(self, name)*(100 if name.upper() else 1):.2f}' for name in self.METRICS)
    
    def improves(self, other: Metric) -> bool:
        assert all(k1 == k2 for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS)) 
        return any(getattr(self, k1) > getattr(other, k2) for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS))

    def save(self, path: str):
        with open(path, 'wb') as writer:
            pickle.dump(self.to('cpu'), writer)
            
    def values(self, scale: float = 1) -> List[float]:
        return [getattr(self, name)*scale for name in self.METRICS]

    def items(self, scale: float = 1) -> Dict[str, float]:
        return {name: getattr(self, name)*scale for name in self.METRICS}.items()
    
    def add_control(self, control):
        for name, value in control.items():
            if name not in self.METRICS + self.ATTRIBUTES:
                self.__setattr__(name, value)
    
    @classmethod
    def load(cls, path: str) -> Metric:
        with open(path, 'rb') as reader:
            m = pickle.load(reader)
        # m = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        return m
    
    def sync(self) -> Metric:
        for name in self.ATTRIBUTES:
            dist.all_reduce(getattr(self, name), op=dist.ReduceOp.SUM)
        return self
            
    def to(self, device: int) -> Metric:
        for name in self.ATTRIBUTES:
            self.__setattr__(name, getattr(self, name).to(device))
        return self 
    
    def copy(self):
        path = mkftemp()
        os.makedirs(folderpath(path), exist_ok=True)
        self.save(path)
        new = Metric.load(path)
        os.remove(path)
        return new 
        
    
