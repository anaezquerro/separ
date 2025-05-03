from __future__ import annotations
from typing import List, Dict
import torch.distributed as dist 
import torch 

from separ.utils.metric.metric import Metric 
from separ.data import TargetTokenizer
from separ.utils.fn import div 



class TaggingMetric(Metric):
    
    def __init__(self, target_tkzs: List[TargetTokenizer]):
        self.METRICS = [tkz.field for tkz in target_tkzs]
        self.KEY_METRICS = self.METRICS
        self.ATTRIBUTES = ['tp', 'npred', 'ngold', 'n'] + \
            [f'_{tkz.field.lower()}' for tkz in target_tkzs]
        for attr in self.ATTRIBUTES[:3]:
            # each one is a dictionary of tensors 
            self.__setattr__(attr, {tkz.field: torch.zeros(len(tkz)) for tkz in target_tkzs})
        self.n = torch.tensor(0)
        for tkz in target_tkzs:
            self.__setattr__(f'_{tkz.field.lower()}', torch.tensor(1e-12))
            
    def __add__(self, other: Metric) -> Metric:
        assert len(set(self.METRICS) - set(other.METRICS)) == 0, f'These two tagging metrics have different fields'
        new = self.copy()
        for attr in new.ATTRIBUTES:
            if attr.startswith('_') or attr == 'n':
                new.__setattr__(attr, new.__getattr__(attr) + getattr(other, attr))
            else:
                d = new.__getattr__(attr) # dictionary 
                for metric in new.METRICS:
                    d[metric] += getattr(other, attr)[metric]
                new.__setattr__(attr, d)
        return new 
    
    def __call__(self, preds: List[torch.Tensor], targets: List[torch.Tensor]) -> TaggingMetric:
        """Update the tagging metric.

        Args:
            preds (List[torch.Tensor] ~ num_fields): Predictions.
            targets (List[torch.Tensor] ~ num_fields): Real tags

        Returns:
            TaggingMetric.
        """
        for field, pred, target in zip(self.METRICS, preds, targets):
            attr = f'_{field.lower()}'
            self.__setattr__(attr, self.__getattr__(attr) + (pred == target).sum().item())
            self.n += len(pred)
            for c in target.unique():
                cpred = pred == c
                cgold = target == c
                self.tp[field][c] += (cgold & cpred).sum().item()
                self.npred[field][c] += cpred.sum().item()
                self.ngold[field][c] += cgold.sum().item() 
        return self 
                
    @property 
    def precision(self) -> Dict[str, Dict[int, float]]:
        return {metric: (self.tp[metric]/self.npred[metric]).fillna(1)*100. for metric in self.METRICS}
    
    @property
    def recall(self) -> Dict[str, Dict[int, float]]:
        return {metric: (self.tp[metric]/self.ngold[metric]).fillna(1)*100. for metric in self.METRICS}
    
    @property
    def fscore(self) -> Dict[str, Dict[int, float]]:
        prec = self.precision
        rec = self.recall 
        return {metric: (2*rec[metric]*prec[metric])/(rec[metric]+prec[metric]) for metric in self.METRICS}
    
    def __getattr__(self, name):
        if name in self.METRICS:
            return div(object.__getattribute__(self, f'_{name.lower()}'), self.n)
        else:
            return object.__getattribute__(self, name)
    
    def to(self, device: int) -> Metric:
        for attr in self.ATTRIBUTES:
            if attr.startswith('_') or attr == 'n':
                self.__setattr__(attr, self.__getattr__(attr).to(device))
            else:
                d = self.__getattr__(attr)
                for metric in self.METRICS:
                    d[metric] = d[metric].to(device)
                self.__setattr__(attr, d)
        return self 
    
    def sync(self):
        for attr in self.ATTRIBUTES:
            if attr.startswith('_') or (attr == 'n'):
                dist.all_reduce(getattr(self, attr), op=dist.ReduceOp.SUM)
            else:
                d = self.__getattr__(attr)
                for metric in self.METRICS:
                    dist.all_reduce(d[metric], op=dist.ReduceOp.SUM)
