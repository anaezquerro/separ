
from __future__ import annotations
from typing import Iterator, Union
from separ.utils.metric.metric import Metric 
from separ.data import Graph
from separ.utils.fn import div 

class SemanticMetric(Metric):
    ATTRIBUTES = ['ur', 'up', 'uf', 'um', 'lr', 'lp', 'lf', 'lm', 'n']
    METRICS = ['UR', 'UP', 'UF', 'UM', 'LR', 'LP', 'LF', 'LM']
    KEY_METRICS = ['UF', 'LF']
    
    def __call__(
        self, 
        preds: Union[Graph, Iterator[Graph]], 
        golds: Union[Graph, Iterator[Graph]],
        **_
    ) -> SemanticMetric:
        if isinstance(preds, Graph):
            self.apply(preds, golds)
        else:
            for pred, gold in zip(preds, golds):
                self.apply(pred, gold)
        return self 
            
    def apply(self, pred: Graph, gold: Graph):
        pred_graph, gold_graph = pred.ADJACENT, gold.ADJACENT 
        mask = (pred_graph & gold_graph)
        tp = mask.sum()
        n_pred, n_gold = pred_graph.sum(), gold_graph.sum() 
        ur, up = div(tp, n_gold), div(tp, n_pred)
        self.uf += div(2*ur*up, ur+up)
        self.ur += ur 
        self.up += up 
        self.um += (pred_graph == gold_graph).all()
        
        # rels = set(gold.rels)
        # lr, lp = 0, 0
        pred_graph, gold_graph = pred.LABELED_ADJACENT, gold.LABELED_ADJACENT
        ltp = ((pred_graph == gold_graph) & mask.numpy()).sum()
        # for rel in rels:
        #     ltp = ((pred_graph == rel) & (gold_graph == rel) & mask.numpy()).sum()
        #     n_pred, n_gold = (pred_graph == rel).sum(), (gold_graph == rel).sum()
        #     lr += div(ltp, n_gold)
        #     lp += div(ltp, n_pred)
        # lr = div(lr, len(rels))
        # lp = div(lp, len(rels))
        lr = div(ltp, n_gold)
        lp = div(ltp, n_pred)
        self.lr += lr 
        self.lp += lp 
        self.lf += div(2*lr*lp, lr+lp)
        self.n += 1 
        self.lm += (pred_graph == gold_graph).all()
        
            
    # def dataset(self, pred: SDP, gold: SDP):
    #     tmp_folder = tempfile.mkdtemp()
        
    #     # SDP evaluation
    #     pred_path = f'{tmp_folder}/pred.{pred.EXTENSION}'
    #     gold_path = f'{tmp_folder}/gold.{gold.EXTENSION}'
    #     pred.save(pred_path)
    #     gold.save(gold_path)
        
    #     result = shell(f'./{SDP_SCRIPT} Scorer {gold_path} {pred_path}')
    #     labeled = re.search("### Labeled scores(.|\n)*?###", result)
    #     unlabeled = re.search("### Unlabeled scores(.|\n)*?###", result)
    #     _, labeled, _ = result[labeled.start():labeled.end()].split('\n\n')
    #     _, unlabeled, _ = result[unlabeled.start():unlabeled.end()].split('\n\n')
    #     labeled, unlabeled = labeled.lower().split('\n'), unlabeled.lower().split('\n')
    #     for item in labeled + unlabeled:
    #         key, value = item.split(': ')
    #         value = float(value.replace(',', '.'))*len(pred)
    #         self.__setattr__(key, getattr(self, key) + (value if not np.isnan(value) else 0.0))
            
    #     self.n += len(pred)
    #     shutil.rmtree(tmp_folder)
        
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