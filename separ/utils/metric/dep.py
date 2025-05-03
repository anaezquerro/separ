from __future__ import annotations
from typing import Iterator, Union 

from separ.utils.metric.metric import Metric 
from separ.data import CoNLL  
from separ.utils.fn import div 
        
class DependencyMetric(Metric):
    METRICS = ['UAS', 'LAS', 'UCM', 'LCM']
    KEY_METRICS = ['UAS', 'LAS']
    ATTRIBUTES = ['uas', 'las', 'ucm', 'lcm', 'n']
    
        
    def __call__(
        self,
        preds: Union[CoNLL.Tree, Iterator[CoNLL.Tree]], 
        golds: Union[CoNLL.Tree, Iterator[CoNLL.Tree]]
    ) -> DependencyMetric:
        if isinstance(preds, CoNLL.Tree):
            self.apply(preds, golds)
        else:
            for pred, gold in zip(preds, golds):
                self.apply(pred, gold)
        return self 
    
    def apply(self, pred: CoNLL.Tree, gold: CoNLL.Tree):
        pred_heads, gold_heads, pred_rels, gold_rels = pred.heads, gold.heads, pred.rels, gold.rels
        mask = pred_heads == gold_heads
        self.uas += mask.mean().item()
        self.las += ((pred_rels == gold_rels) & mask).mean().item()
        self.ucm += mask.all().item()
        self.lcm += ((pred_rels == gold_rels) & mask).all().item()
        self.n += 1
        
    @property
    def UAS(self) -> float:
        return div(self.uas, self.n)
        
    @property
    def LAS(self) -> float:
        return div(self.las, self.n)
    
    @property
    def UCM(self) -> float:
        return div(self.ucm, self.n)
    
    @property
    def LCM(self) -> float:
        return div(self.lcm, self.n)
        

