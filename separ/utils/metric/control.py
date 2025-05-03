from __future__ import annotations
import torch 

from separ.utils.metric.metric import Metric 
from separ.utils.fn import div 

class ControlMetric(Metric):
    
    def __init__(self, **control):
        self.ATTRIBUTES = ['n']
        self.METRICS = []
        self.n = torch.tensor(int(len(control) > 0))
        for name, value in control.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            self.__setattr__(f'_{name.lower()}', value)
            self.ATTRIBUTES.append(f'_{name.lower()}')
            self.METRICS.append(name)
        
    def __add__(self, other: ControlMetric) -> ControlMetric:
        new = self.copy()
        # intersection 
        for attr in set(new.ATTRIBUTES) & set(other.ATTRIBUTES):
            new.__setattr__(attr, getattr(new, attr) + getattr(other, attr))
        # difference 
        for attr in set(other.ATTRIBUTES) - set(new.ATTRIBUTES):
            new.__setattr__(attr, getattr(other, attr))
        new.METRICS += [m for m in other.METRICS if m not in new.METRICS]
        new.ATTRIBUTES += [a for a in other.ATTRIBUTES if a not in new.ATTRIBUTES]
        return new
    
    def __repr__(self):
        return f', '.join(f'{name}={getattr(self, name):.2f}' for name in self.METRICS)
    
    def __getattr__(self, name):
        if name in self.METRICS:
            return div(object.__getattribute__(self, f'_{name.lower()}'), self.n)
        else:
            return object.__getattribute__(self, name)
        
