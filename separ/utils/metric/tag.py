from __future__ import annotations
import torch 

from separ.utils.metric.metric import Metric 
from separ.data import TargetTokenizer
from separ.utils.fn import div 
from separ.utils.shard import is_distributed, sync



class TaggingMetric(Metric):
    
    def __init__(self, target_tkzs: list[TargetTokenizer]):
        self.METRICS = [tkz.field for tkz in target_tkzs]
        self.KEY_METRICS = self.METRICS
        self.ATTRIBUTES = ['tp', 'npred', 'ngold', 'n'] + \
            [f'_{tkz.field.lower()}' for tkz in target_tkzs]
        for attr in self.ATTRIBUTES[:3]:
            # each one is a dictionary of tensors 
            self.__setattr__(attr, {tkz.field: torch.zeros(len(tkz)) for tkz in target_tkzs})
        self.n = torch.zeros(len(target_tkzs))
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
    
    def __call__(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> TaggingMetric:
        """Update the tagging metric.

        Args:
            preds (List[torch.Tensor] ~ num_fields): Predictions.
            targets (List[torch.Tensor] ~ num_fields): Real tags

        Returns:
            TaggingMetric.
        """
        for i, (field, pred, target) in enumerate(zip(self.METRICS, preds, targets)):
            attr = f'_{field.lower()}'
            self.__setattr__(attr, self.__getattr__(attr) + (pred == target).sum().item())
            self.n[i] += len(pred)
            for c in target.unique():
                cpred = pred == c
                cgold = target == c
                self.tp[field][c] += (cgold & cpred).sum().item()
                self.npred[field][c] += cpred.sum().item()
                self.ngold[field][c] += cgold.sum().item() 
        return self 
                
    @property 
    def precision(self) -> dict[str, dict[int, float]]:
        return {metric: div(self.tp[metric], self.npred[metric])*100. for metric in self.METRICS}
    
    @property
    def recall(self) -> dict[str, dict[int, float]]:
        return {metric: div(self.tp[metric], self.ngold[metric])*100. for metric in self.METRICS}
    
    @property
    def fscore(self) -> dict[str, dict[int, float]]:
        prec = self.precision
        rec = self.recall 
        return {metric: (2*rec[metric]*prec[metric])/(rec[metric]+prec[metric])
                if (rec[metric]+prec[metric]) > 0 else 0
                for metric in self.METRICS}
    
    def __getattr__(self, name) -> float:
        if name in self.METRICS:
            return float(div(object.__getattribute__(self, f'_{name.lower()}'), self.n[self.METRICS.index(name)]))
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
    
    def sync(self, rank: int | None = None):
        if is_distributed():
            for attr in self.ATTRIBUTES:
                if attr.startswith('_') or (attr == 'n'):
                    sync(getattr(self, attr), rank)
                else:
                    d = self.__getattr__(attr)
                    for metric in self.METRICS:
                        sync(d[metric], rank)
        return self

    @classmethod
    def load(self, path: str) -> TaggingMetric:
        m = super().load(path)
        if m.n.ndim == 0:
            m.n = m.n.expand(len(m.METRICS))
        return m