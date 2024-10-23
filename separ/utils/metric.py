from __future__ import annotations
from typing import List, Dict, Union, Optional
import re, shutil, subprocess, pickle, tempfile
import numpy as np
import torch 

from separ.utils.fn import init_folder, parallel, div, avg
from separ.data import SDP, CoNLL,  PTB
from separ.utils.common import SDP_SCRIPT, CON_SCRIPT

class Metric:
    METRICS = [] 
    ATTRIBUTES = []
    KEY_METRICS = []
    
    def __init__(self, *args, **kwargs):
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, 0.0)
        self._control = dict() # other control metrics 
        
        if len(args) > 0 or len(kwargs) > 0:
            self(*args, **kwargs)
    
    def __eq__(self, other: Metric) -> bool:
        return isinstance(other, self.__class__) and all(getattr(self, attr) == getattr(other, attr) for attr in self.ATTRIBUTES)
    
    def __add__(self, other: Metric) -> Metric:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        return self
    
    def __radd__(self, other: Metric) -> Metric:
        return self + other if isinstance(other, Metric) else self 
    
    def __call__(self, *args, **kwargs) -> Metric:
        raise NotImplementedError
    
    def __repr__(self):
        return f', '.join(f'{name.upper()}={round(float(getattr(self, name)*100), 2)}' for name in self.METRICS)
    
    def __getitem__(self, name: str) -> float:
        return self.__getattribute__(name)
    
    def improves(self, other: Metric) -> bool:
        assert all(k1 == k2 for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS)) 
        return any(getattr(self, k1) > getattr(other, k2) for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS))

    def save(self, path: str):
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)
            
    def values(self, scale: float = 1) -> List[float]:
        return [getattr(self, name)*scale for name in self.METRICS]

    def items(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.METRICS}
    
    def add_control(self, **control):
        for name, value in control.items():
            self.__setattr__(name, value)
    
    @classmethod
    def load(cls, path: str) -> Metric:
        with open(path, 'rb') as reader:
            metric = pickle.load(reader)
        return metric 

class SemanticMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'uf', 'um', 'lr', 'lp', 'lf', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['UF', 'LF']
    sdp_script = SDP_SCRIPT
    
    def __call__(
        self, 
        pred: Union[SDP, List[SDP.Graph], SDP.Graph], 
        gold: Union[SDP, List[SDP.Graph], SDP.Graph],
        num_workers: int = 1
    ) -> SemanticMetric:
        """Compute SDP evaluation metric.
        - If the input data are SDP graphs, it maually extracts the official metrics.
        - If the input data are SDP datasets, it calls the sdp-eval script.

        Args:
            pred (Union[SDP, SDP.Graph]): Predicted dataset/graph.
            gold (Union[SDP, SDP.Graph]): Ground truth dataset/graph.
            tmp_folder (Optional[str], optional): Temporal folder. Defaults to .tmp/.
            num_workers (int, optional): Number of workers to parallelize computation.
            
        Raises:
            NotImplementedError: If prediction and ground truth inputs are not SDP datasets or graphs.

        Returns:
            SemanticMetric: SDP evaluation.
        """
        if isinstance(pred, list):
            pred = SDP(pred, 'pred')
        if isinstance(gold, list):
            gold = SDP(gold, 'gold')
        if isinstance(pred, SDP.Graph) and isinstance(gold, SDP.Graph):
            self.graph(pred, gold)
        elif isinstance(pred, SDP) and isinstance(gold, SDP):
            if self.sdp_script:
                self.dataset(pred, gold)
            else:
                self += sum(parallel(SemanticMetric, pred, gold, name='sdp-metric'))
        return self 
            
    def graph(self, pred: SDP.Graph, gold: SDP.Graph):
        pred_graph, gold_graph = pred.ADJACENT, gold.ADJACENT 
        tp = (pred_graph & gold_graph).sum()
        n_pred, n_gold = pred_graph.sum(), gold_graph.sum() 
        ur, up = div(tp, n_gold), div(tp, n_pred)
        self.uf += div(2*ur*up, ur+up)
        self.ur += ur 
        self.up += up 
        self.um += (pred_graph == gold_graph).all()
        
        rels = gold.rels
        lr, lp = 0, 0
        pred_graph, gold_graph = pred.LABELED_ADJACENT, gold.LABELED_ADJACENT
        for rel in rels:
            ltp = ((pred_graph == rel) & (gold_graph == rel)).sum()
            n_pred, n_gold = (pred_graph == rel).sum(), (gold_graph == rel).sum()
            lr += div(ltp, n_gold)
            lp += div(ltp, n_pred)
        lr = div(lr, len(rels))
        lp = div(lp, len(rels))
        self.lr += lr 
        self.lp += lp 
        self.lf += div(2*lr*lp, lr+lp)
        self.n += 1 
        self.lm += (pred_graph == gold_graph).all()
        
            
    def dataset(self, pred: SDP, gold: SDP):
        tmp_folder = tempfile.mkdtemp()
        
        # SDP evaluation
        pred_path = f'{tmp_folder}/pred.{pred.EXTENSION}'
        gold_path = f'{tmp_folder}/gold.{gold.EXTENSION}'
        pred.save(pred_path)
        gold.save(gold_path)
        
        result = subprocess.check_output(f'./{self.sdp_script} Scorer {gold_path} {pred_path}', shell=True, text=True, stderr=subprocess.STDOUT)
        labeled = re.search("### Labeled scores(.|\n)*?###", result)
        unlabeled = re.search("### Unlabeled scores(.|\n)*?###", result)
        _, labeled, _ = result[labeled.start():labeled.end()].split('\n\n')
        _, unlabeled, _ = result[unlabeled.start():unlabeled.end()].split('\n\n')
        labeled, unlabeled = labeled.lower().split('\n'), unlabeled.lower().split('\n')
        for item in labeled + unlabeled:
            key, value = item.split(': ')
            value = float(value.replace(',', '.'))*len(pred)
            self.__setattr__(key, getattr(self, key) + (value if not np.isnan(value) else 0.0))
            
        self.n += len(pred)
        shutil.rmtree(tmp_folder)
        
    @property
    def UR(self):
        return div(self.ur, self.n)
    
    @property
    def UP(self):
        return div(self.up, self.n)
    
    @property 
    def UF(self):
        return div(self.uf, self.n)
    
    @property 
    def UM(self):
        return div(self.um, self.n)
    
    @property
    def LR(self):
        return div(self.lr, self.n)
    
    @property
    def LP(self):
        return div(self.lp, self.n)
    
    @property 
    def LF(self):
        return div(self.lf, self.n)
    
    @property 
    def LM(self):
        return div(self.lm, self.n)

    
    
class DependencyMetric(Metric):
    ATTRIBUTES = ['uas', 'las', 'ucm', 'lcm', 'n']
    METRICS = ['UAS', 'LAS', 'UCM', 'LCM']
    KEY_METRICS = ['LAS']
    
    def __call__(
            self, 
            pred: CoNLL,  
            gold: CoNLL,  
            num_workers: int = 1,
            **_
        ) -> DependencyMetric:
        if isinstance(pred, CoNLL) and isinstance(gold, CoNLL):
            metric = sum(parallel(DependencyMetric, pred, gold, num_workers=num_workers, name='dep-metric'))
            self.__add__(metric)
        elif isinstance(pred, CoNLL.Graph) and isinstance(gold, CoNLL.Graph):
            self.apply(pred, gold)
        else:
            raise NotImplementedError
        return self
        
    def apply(self, pred: CoNLL.Graph, gold: CoNLL.Graph) -> DependencyMetric:
        pred_heads, gold_heads, pred_rels, gold_rels = map(np.array, [pred.HEAD, gold.HEAD, pred.DEPREL, gold.DEPREL])
        umask = pred_heads == gold_heads
        lmask = umask & (pred_rels == gold_rels)
        self.uas += umask.mean()
        self.las += lmask.mean()
        self.ucm += umask.all()
        self.lcm += lmask.all()
        self.n += 1
        return self 
        
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
    
    
class ConstituencyMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'uf', 'um', 'lr', 'lp', 'lf', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['LF']
    con_script = CON_SCRIPT
    
    def __call__(
        self, 
        pred: Union[PTB, PTB.Tree],
        gold: Union[PTB, PTB.Tree],
        num_workers: int = 1
    ) -> ConstituencyMetric:
        if isinstance(pred, PTB) and isinstance(gold, PTB):
            if self.con_script:
                self.dataset(pred, gold, num_workers)
            else:
                self += sum(parallel(ConstituencyMetric, pred, gold, num_workers=num_workers, name='con-metric'))
        elif isinstance(pred, PTB.Tree) and isinstance(gold, PTB.Tree):
            return self.apply(pred, gold)
        else:
            raise ValueError
        return self 
        
    def dataset(self, pred: PTB, gold: PTB) -> ConstituencyMetric:
        tmp_folder = tempfile.mkdtemp()
        
        gold_path = f'{tmp_folder}/gold.ptb'
        pred_path = f'{tmp_folder}/pred.ptb'
        gold.save(gold_path)
        pred.save(pred_path)
        result = subprocess.check_output(f'{self.con_script} {gold_path} {pred_path}', shell=True, text=True)
        result = result.split('-- All --')[1].split('-- len<=40 --')[0].strip().split('\n')
        lr, lp, lf, lm = [float(line.split('=')[1])/100*len(pred) for line in result[4:8]] 
        self.lr += lr 
        self.lp += lp 
        self.lf += lf 
        self.lm += lm
        
        # unlabeled metrics must be computed manually
        umetric = sum(parallel(ConstituencyMetric, pred, gold, num_workers=1, name='con-metric'))
        self.ur += umetric.ur
        self.up += umetric.up
        self.uf += umetric.uf
        self.um += umetric.um
        self.n += len(pred)
        shutil.rmtree(tmp_folder)
        
    def apply(self, pred: PTB.Tree, gold: PTB.Tree) -> ConstituencyMetric:
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
        
        
class TaggingMetric(Metric):
    KEY_METRICS = ['ACC']
    
    def __init__(self, fields: List[str]):
        self.ATTRIBUTES = [f'_{field.lower()}' for field in fields] + ['n']
        self.METRICS = [field.upper() for field in fields]
        for name in self.METRICS + self.ATTRIBUTES:
            self.__setattr__(name, 0.0)
    
    def __call__(
        self, 
        pred: Optional[Union[CoNLL, PTB, SDP, CoNLL.Graph, PTB.Tree, SDP.Graph]] = None, 
        gold: Optional[Union[CoNLL, PTB, SDP, CoNLL.Graph, PTB.Tree, SDP.Graph]] = None, 
        **_
    ) -> TaggingMetric:
        if pred is None and gold is None:
            return self 
        elif isinstance(pred, CoNLL) or isinstance(pred, PTB) or isinstance(pred, SDP):
            self.dataset(pred, gold)
        else:
            self.apply(pred, gold)
        return self
    
    def dataset(self, pred: Union[CoNLL, PTB, SDP], gold: Union[CoNLL, PTB, SDP]):
        self.n += pred.n_tokens 
        for metric in self.METRICS:
            attr = f'_{metric.lower()}' 
            self.__setattr__(attr, self.__getattribute__(attr) + \
                (torch.cat([getattr(p, metric) for p in pred]) == torch.cat([getattr(g, metric) for g in gold])).sum().item()
            )
            self.__setattr__(metric, self.__getattribute__(attr)/self.n)

    def apply(self, pred: Union[CoNLL.Graph, PTB.Tree, SDP.Graph], gold: Union[CoNLL.Graph, PTB.Tree, SDP.Graph]):
        self.n += len(pred)
        for metric in self.METRICS:
            attr = f'_{metric.lower()}'
            self.__setattr__(attr, self.__getattribute__(attr) + (getattr(pred, metric) == getattr(gold, metric)).sum().item())
            self.__setattr__(metric, self.__getattribute__(attr)/self.n)

    @property
    def ACC(self) -> float:
        return avg(getattr(self, metric) for metric in self.METRICS)

        
        