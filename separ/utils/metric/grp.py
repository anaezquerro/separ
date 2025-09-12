
from __future__ import annotations
import re, os
import numpy as np 

from separ.utils.metric.metric import Metric 
from separ.data import Graph, SDP, EnhancedCoNLL, Dataset
from separ.utils.fn import div, mkftemp, shell, is_float

class GraphMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'um', 'lr', 'lp', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['UF', 'LF']
    
    @staticmethod
    def remove_empty(path: str):
        lines = map(str.strip, open(path, 'r').read().split('\n'))
        lines = [line for line in lines if not (len(line) > 0 and is_float(line.split()[0]))]
        with open(path, 'w') as writer:
            writer.write('\n'.join(lines))
            writer.write('\n\n') # add extra lines
    
    def __init__(self, SCRIPT: str | None = None):
        self.SCRIPT = SCRIPT 
        super().__init__()
    
    def __call__(
        self, 
        pred: Graph | list[Graph] | SDP | EnhancedCoNLL, 
        gold: Graph | list[Graph] | SDP | EnhancedCoNLL,
        **_
    ) -> GraphMetric:
        if isinstance(pred, Graph) and isinstance(gold, Graph):
            self.apply(pred, gold)
        elif self.SCRIPT and isinstance(pred, Dataset) and isinstance(gold, Dataset):
            self.official(pred, gold)
        else:
            for pred, gold in zip(pred, gold):
                self.apply(pred, gold)
        return self 
            
    def apply(self, pred: Graph, gold: Graph):
        pred_graph, gold_graph = pred.ADJACENT, gold.ADJACENT 
        mask = (pred_graph & gold_graph)
        tp = mask.sum()
        n_pred, n_gold = pred_graph.sum(), gold_graph.sum() 
        ur, up = div(tp, n_gold), div(tp, n_pred)
        self.ur += ur 
        self.up += up 
        self.um += (pred_graph == gold_graph).all()
        
        pred_graph, gold_graph = pred.LABELED_ADJACENT, gold.LABELED_ADJACENT
        ltp = ((pred_graph == gold_graph) & mask.numpy()).sum()
        lr = div(ltp, n_gold)
        lp = div(ltp, n_pred)
        self.lr += lr 
        self.lp += lp 
        self.n += 1 
        self.lm += (pred_graph == gold_graph).all()
            
    def official(self, pred: SDP | EnhancedCoNLL, gold: SDP | EnhancedCoNLL):
        if isinstance(pred, EnhancedCoNLL):
            pred = pred.to_sdp()
        if isinstance(gold, EnhancedCoNLL):
            gold = gold.to_sdp()
            
        pred_path = f'{mkftemp()}.{pred.EXTENSION}'
        gold_path = f'{mkftemp()}.{gold.EXTENSION}'
        pred.save(pred_path)
        gold.save(gold_path)
        
        result = shell(f'./{self.SCRIPT} Scorer {gold_path} {pred_path}').stderr.strip()
        labeled = re.search("### Labeled scores(.|\n)*?###", result)
        unlabeled = re.search("### Unlabeled scores(.|\n)*?###", result)
        _, labeled, _ = result[labeled.start():labeled.end()].split('\n\n')
        _, unlabeled, _ = result[unlabeled.start():unlabeled.end()].split('\n\n')
        labeled, unlabeled = labeled.lower().split('\n'), unlabeled.lower().split('\n')
        up, ur, _, um = [float(item.split(': ')[1]) for item in unlabeled]
        lp, lr, _, lm = [float(item.split(': ')[1]) for item in labeled]
        
        self.ur += ur*len(pred)
        self.up += up*len(pred)
        self.um += um*len(pred)
        self.lr += lr*len(pred)
        self.lp += lp*len(pred)
        self.lm += lm*len(pred)
        self.n += len(pred)
        os.remove(pred_path)
        os.remove(gold_path)
        
    @property
    def UR(self) -> float:
        return float(div(self.ur, self.n))
    
    @property
    def UP(self) -> float:
        return float(div(self.up, self.n))
    
    @property
    def UF(self) -> float:
        UR, UP = self.UR, self.UP 
        return (2*UR*UP)/(UR+UP) if (UR + UP) > 0 else 0
    
    @property
    def UM(self) -> float:
        return float(div(self.um, self.n))
        
    @property
    def LR(self) -> float:
        return float(div(self.lr, self.n))
    
    @property
    def LP(self) -> float:
        return float(div(self.lp, self.n))
    
    @property
    def LF(self) -> float:
        LR, LP = self.LR, self.LP
        return (2*LR*LP)/(LR+LP) if (LR + LP) > 0 else 0
    
    @property 
    def LM(self) -> float:
        return float(div(self.lm, self.n))