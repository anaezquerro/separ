from __future__ import annotations
import os 

from separ.utils.metric.metric import Metric 
from separ.utils.fn import div, mkftemp, shell
from separ.data import PTB 
    
    
class ConstituencyMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'um', 'lr', 'lp', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['LF']
    
    def __init__(
        self,
        SCRIPT: str | None = None,
        LABELED_PRM: str | None = None,
        UNLABELED_PRM: str | None = None 
    ):
        self.SCRIPT = SCRIPT 
        self.LABELED_PRM = LABELED_PRM
        self.UNLABELED_PRM = UNLABELED_PRM
        super().__init__()
    
    def __call__(
        self, 
        pred: PTB.Tree | list[PTB.Tree] | PTB,
        gold: PTB.Tree | list[PTB.Tree] | PTB
    ) -> ConstituencyMetric:
        if isinstance(pred, PTB.Tree) and isinstance(gold, PTB.Tree):
            self.apply(pred, gold)
        elif isinstance(pred, PTB) and isinstance(gold, PTB):
            self.official(pred, gold)
        else:
            for p, g in zip(pred, gold):
                self.apply(p, g)
        return self 
        
    def official(self, pred: PTB, gold: PTB, **_) -> ConstituencyMetric:
        pred.path = f'{mkftemp()}.{pred.EXTENSION}'
        pred.save(pred.path)
        
        # compute unlabeled metrics
        result = shell(f'{self.SCRIPT} -p {self.UNLABELED_PRM} {gold.path} {pred.path}').stdout.strip()
        result = result.split('-- All --')[1].split('-- len<=40 --')[0].strip().split('\n')
        ur, up, _, um = [float(line.split('=')[1]) for line in result[4:8]] 
        self.ur += ur/100*len(pred) 
        self.up += up/100*len(pred)
        self.um += um/100*len(pred)
        
        # compute labeled metrics 
        result = shell(f'./{self.SCRIPT} -p {self.LABELED_PRM} {gold.path} {pred.path}').stdout.strip()
        result = result.split('-- All --')[1].split('-- len<=40 --')[0].strip().split('\n')
        lr, lp, _, lm = [float(line.split('=')[1])for line in result[4:8]] 
        self.lr += lr/100*len(pred)
        self.lp += lp/100*len(pred)
        self.lm += lm/100*len(pred)
        
        # update total
        self.n += len(pred)
        os.remove(pred.path)
        return self 
        
    def apply(self, pred: PTB.Tree, gold: PTB.Tree):
        umask = pred.MATRIX & gold.MATRIX 
        lmask = pred.LABELED_MATRIX == gold.LABELED_MATRIX
        ur, up = div(umask.sum(), gold.MATRIX.sum()), div(umask.sum(), pred.MATRIX.sum())
        self.ur += ur 
        self.up += up 
        self.um += (pred.MATRIX == gold.MATRIX).all().item()
        ltp = (pred.LABELED_MATRIX[umask] == gold.LABELED_MATRIX[umask]).sum()
        lr = div(ltp, gold.MATRIX.sum())
        lp = div(ltp, pred.MATRIX.sum())
        self.lr += lr
        self.lp += lp
        self.lm += lmask.all()
        self.n += 1
        return self 
    
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
        
        