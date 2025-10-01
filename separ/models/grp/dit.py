from __future__ import annotations
from torch import nn
import torch 

from separ.models.grp.sl import GraphSLParser
from separ.models.grp.model import GraphSLModel
from separ.models.tag.dit import DiTagger
from separ.models.tag.dit.model import DiTModel
from separ.utils import Config, ControlMetric, pad2D, avg, get_mask
from separ.modules import FFN 
from separ.data import Graph, adjacent_from_arcs, SDP, EnhancedCoNLL, PretrainedTokenizer

class GraphDiTModel(DiTModel, GraphSLModel):
    def __init__(
        self,
        enc_conf: Config,
        word_conf: Config,
        dec_conf: Config,
        label_conf: Config,
        rel_conf: Config
    ):
        self.label_conf = label_conf
        self.rel_conf = rel_conf
        if rel_conf.join_rels:
            DiTModel.__init__(self, enc_conf, word_conf, dec_conf, label_conf, rel_conf)
        else:
            DiTModel.__init__(self, enc_conf, word_conf, dec_conf, label_conf)
            self.rel_criteria = nn.CrossEntropyLoss()
            self.head = FFN(self.hidden_size, self.hidden_size//2)
            self.dep = FFN(self.hidden_size, self.hidden_size//2)
            self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
            
    def forward(
        self,
        words: torch.Tensor,
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ):
        if self.rel_conf.join_rels:
            return DiTModel.forward(self, words, masks, targets)
        else:
            labels, rels, mask, matrices = *targets, *masks 
            embed = self.encode(words)
            preds, signals = map(list, self.decode(embed[:, 1:], [mask], [labels]))
            s_rel = self.score_rel(embed, matrices, int(mask.sum()))
            preds.append(s_rel)
            signals.append(rels)
            return preds, signals
        
    def loss(self, preds: list[torch.Tensor], targets: list[torch.Tensor]):
        if self.rel_conf.join_rels:
            return super().loss(preds, targets)
        else:
            return super().loss(preds[:-1], targets[:-1]) + self.rel_criteria(preds[-1], targets[-1])

    def predict_label(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = self.__getattr__(self.label_conf.name.lower()).predict(embed, mask)
        pred[pred >= self.label_conf.vocab_size] = self.label_conf.most_frequent 
        for i in self.label_conf.special_indices:
            pred[pred == i] = self.label_conf.most_frequent
        return pred
    

class GraphDiTParser(GraphSLParser, DiTagger):
    MODEL = GraphDiTModel

    def collate(self, graphs: list[Graph]):
        inputs = [tkz.batch_encode(graphs, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(graphs, pin=False) for tkz in self.target_tkzs]
        mask = get_mask(list(map(len, graphs)))
        if self.join_rels:
            masks = [mask, mask]
        else:
            targets[-1] = self.REL.batch_encode(graphs, mode='cat', pin=False)
            masks = [mask, pad2D([graph.MATRIX for graph in graphs])]
        return inputs, masks, targets, graphs
    
    def train_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        return DiTagger.train_step(self, inputs, masks, targets)
        
    def _pred_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        graphs: list[Graph]
    ) -> tuple[list[Graph], float]:
        mask, _ = masks
        lens = mask.sum(-1).tolist()
        if self.join_rels:
            label_preds, rel_preds = self.model.predict(*inputs, masks)
            preds, well_formed = zip(*map(self._pred, graphs, label_preds.split(lens), rel_preds.split(lens)))
        else:
            embed = self.model.encode(*inputs)
            label_preds = self.model.predict_label(embed[:, 1:], mask)
            arc_preds, well_formed = zip(*map(self._pred_label, label_preds.split(lens)))
            matrix_preds = pad2D(map(adjacent_from_arcs, arc_preds, lens))
            num_arcs = matrix_preds.sum((-2,-1)).tolist()
            rel_preds = self.model.predict_rel(embed, matrix_preds, masks[0].sum())
            preds = map(self._pred_rel, graphs, arc_preds, rel_preds.split(num_arcs))
        return list(preds), avg(well_formed)*100
    
    @classmethod
    def build(
        cls,
        data: str | SDP | EnhancedCoNLL,
        enc_conf: Config,
        word_conf: Config,
        dec_conf: Config,
        join_rels: bool = False,
        **_
    ) -> GraphDiTParser:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM', bos=not join_rels)
        word_conf |= word_tkz.conf
        
        target_tkzs, target_confs = cls.build_targets(cls.Laeler(**_), data, join_rels)
        return cls(
            input_tkzs=[word_tkz], 
            target_tkzs=target_tkzs, 
            model_confs=[enc_conf, word_conf, dec_conf] + target_confs, 
            join_rels=join_rels, 
            **_
        )