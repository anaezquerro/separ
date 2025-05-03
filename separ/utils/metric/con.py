from __future__ import annotations
from typing import Union, Iterator
import numpy as np

from separ.utils.metric.metric import Metric 
from separ.utils.fn import div 
from separ.data import PTB 
    
    
class ConstituencyMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'uf', 'um', 'lr', 'lp', 'lf', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['UF', 'LF']
    
    def __call__(
        self, 
        preds: Union[PTB.Tree, Iterator[PTB.Tree]],
        golds: Union[PTB.Tree, Iterator[PTB.Tree]]
    ) -> ConstituencyMetric:
        if isinstance(preds, PTB.Tree):
            self.apply(preds, golds)
        else:
            for pred, gold in zip(preds, golds):
                self.apply(pred, gold)
        return self 
        
    # def dataset(self, pred: PTB, gold: PTB, num_workers: int, **_) -> ConstituencyMetric:
    #     tmp_folder = tempfile.mkdtemp()
    #     init_folder(tmp_folder)
    #     gold_path = f'{tmp_folder}/gold.ptb'
    #     pred_path = f'{tmp_folder}/pred.ptb'
    #     gold.save(gold_path)
    #     pred.save(pred_path)
    #     result = shell(f'{CON_SCRIPT} {gold_path} {pred_path}')
    #     result = result.split('-- All --')[1].split('-- len<=40 --')[0].strip().split('\n')
    #     lr, lp, lf, lm = [float(line.split('=')[1])/100*len(pred) for line in result[4:8]] 
    #     self.lr += lr 
    #     self.lp += lp 
    #     self.lf += lf 
    #     self.lm += lm
        
    #     umetric = sum(parallel(ConstituencyMetric, pred, gold, num_workers=num_workers, name='con-metric'))
    #     self.ur += umetric.ur
    #     self.up += umetric.up
    #     self.uf += umetric.uf
    #     self.um += umetric.um
    #     self.n += len(pred)
    #     shutil.rmtree(tmp_folder)
    #     return self 
        
    def apply(self, pred: PTB.Tree, gold: PTB.Tree):
        umask = pred.MATRIX & gold.MATRIX 
        lmask = pred.LABELED_MATRIX == gold.LABELED_MATRIX
        ur, up = div(umask.sum(), gold.MATRIX.sum()), div(umask.sum(), pred.MATRIX.sum())
        self.ur += ur 
        self.up += up 
        self.uf += div(2*ur*up, ur+up)
        self.um += (pred.MATRIX == gold.MATRIX).all().item()
        lr, lp = [], []
        for label in np.unique(gold.LABELED_MATRIX):
            ltp = lmask[gold.LABELED_MATRIX == label].sum().item()
            lr.append(div(ltp, (gold.LABELED_MATRIX == label).sum()))
            lp.append(div(ltp, (pred.LABELED_MATRIX == label).sum()))
        lr, lp = map(np.mean, (lr, lp))
        self.lr += lr
        self.lp += lp
        self.lf += div(2*lr*lp, lr+lp)
        self.lm += lmask.all()
        self.n += 1
        return self 
    
    @property
    def UR(self) -> float:
        return div(self.ur, self.n)
    
    @property
    def UP(self) -> float:
        return div(self.up, self.n)
    
    @property
    def UF(self) -> float:
        return div(self.uf, self.n)
    
    @property
    def UM(self) -> float:
        return div(self.um, self.n)
        
    @property
    def LR(self) -> float:
        return div(self.lr, self.n)
    
    @property
    def LP(self) -> float:
        return div(self.lp, self.n)
    
    @property
    def LF(self) -> float:
        return div(self.lf, self.n)
    
    @property 
    def LM(self) -> float:
        return div(self.lm, self.n)
        
        