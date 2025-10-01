
from __future__ import annotations
import torch 

from separ.models.grp.sl import GraphSLParser
from separ.models.grp.model import GraphSLModel
from separ.models.tag.gat.tagger import GaTagger
from separ.models.tag.gat.model import GATModel
from separ.data import Graph, adjacent_from_arcs, SDP, EnhancedCoNLL
from separ.modules import FFN
from separ.utils import get_mask, pad2D, ControlMetric, Config, acc, avg

class GraphGaTModel(GraphSLModel, GATModel):
    
    def __init__(
        self,
        gen_conf: Config,
        dis_conf: Config,
        label_conf: Config,
        rel_conf: Config
    ):
        self.rel_conf = rel_conf 
        self.label_conf = label_conf
        if rel_conf.join_rels:
            GATModel.__init__(self, gen_conf, dis_conf, label_conf, rel_conf)
        else:
            GATModel.__init__(self, gen_conf, dis_conf, label_conf)
            self.hidden_size = self.G_encoder.embed_size
            self.head = FFN(self.hidden_size, self.hidden_size//2)
            self.dep = FFN(self.hidden_size, self.hidden_size//2)
            self.rel = FFN(self.hidden_size, rel_conf.vocab_size)
            
    def forward(
        self,
        words: torch.Tensor,
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.rel_conf.join_rels:
            return GATModel.forward(self, words, masks, targets)
        else:
            labels, rels, mask, matrices = *targets, *masks
            G_embed = self.G_encoder(words)
            D_embed = self.D_word_embed(words)
            loss, scores = GATModel.decode(self, G_embed[:, 1:], D_embed[:, 1:], [mask], [labels])
            s_rel = self.score_rel(G_embed, matrices, int(mask.sum()))
            scores.append(s_rel)
            loss[0] += self.criteria(s_rel, rels)
            return loss, scores 
        
    def predict(self, words: torch.Tensor, masks: list[torch.Tensor]):
        return GATModel.predict(self, words, masks)
        
    def predict_label(self, embed: torch.Tensor, masks: list[torch.Tensor] ) -> list[torch.Tensor]:
        scores = [self.__getattr__(f'G_{conf.name.lower()}')(embed[mask]) for conf, mask in zip(self.target_confs, masks)]
        return self.postprocess(scores)[0]

class GraphGaTParser(GraphSLParser, GaTagger):
    MODEL = GraphGaTModel
    
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
        labels, rels, mask, _ = *targets, *masks
        loss, scores = self.model(*inputs, masks, targets)
        targets = [labels[mask], rels[mask] if self.join_rels else rels]
        return loss, ControlMetric(
            g_loss=loss[0].detach(), d_loss=loss[1].detach(), loss=sum(loss).detach(),
            **dict(zip(self.TARGET_FIELDS, map(acc, scores, targets)))
        )
            
    @torch.no_grad()
    def _pred_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        graphs: list[torch.Tensor]
    ) -> tuple[list[Graph], float]:
        if self.join_rels:
            preds = self.model.predict(inputs[0], masks)
            preds, well_formed = zip(*map(self._pred, graphs, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        else:
            mask, lens = masks[0], masks[0].sum(-1).tolist()
            embed = self.model.G_encoder(*inputs)
            label_preds = self.model.predict_label(embed[:, 1:], masks)
            arc_preds, well_formed = zip(*map(self._pred_label, label_preds.split(lens)))
            matrix_preds = pad2D(map(adjacent_from_arcs, arc_preds, lens))
            num_arcs = matrix_preds.sum((-2,-1)).tolist()
            rel_preds = self.model.predict_rel(embed, matrix_preds, mask.sum()).split(num_arcs)
            preds = map(self._pred_rel, graphs, arc_preds, rel_preds)
        return list(preds), avg(well_formed)*100

    @classmethod
    def build(
        cls,
        data: str | SDP | EnhancedCoNLL,
        word_conf: Config,
        dis_conf: Config,
        join_rels: bool = False,
        **_
    ) -> GraphGaTParser:
        if isinstance(data, str):
            data = cls.load_data(data)

        input_tkzs, in_confs = cls.build_inputs(join_rels, data, word_conf)
        dis_conf |= in_confs[0]
        
        target_tkzs, target_confs, root_rel = cls.build_targets(cls.Labeler(**_), data)
        
        return cls(
            input_tkzs=input_tkzs,
            target_tkzs=target_tkzs, 
            model_confs=[in_confs[0], dis_conf] + target_confs, 
            join_rels=join_rels,
            root_rel=root_rel,
            **_
        )