from __future__ import annotations 
from typing import List, Tuple, Union, Optional, Dict, Iterator 
from argparse import ArgumentParser
import torch 

from trasepar.parser import Parser 
from trasepar.models.dep.labeler import DependencyLabeler
from trasepar.models.dep.pos.model import PoSDependencyModel
from trasepar.utils import Config, flatten, parallel, create_mask, avg, acc, DependencyMetric
from trasepar.data import CoNLL, AbstractTokenizer, Tokenizer, PretrainedTokenizer, CharacterTokenizer
from trasepar.structs import Arc, forms_cycles

class PoSDependencyParser(Parser):
    """PoS-tag Dependency Parser from [Strzyz et al., 2019](https://aclanthology.org/N19-1077/)."""
    NAME = 'dep-pos'
    MODEL = PoSDependencyModel
    DATASET = CoNLL
    METRIC = DependencyMetric
    PARAMS = ['gold']
    
    class Labeler(DependencyLabeler):
        SEP = '$'
        DEFAULT = 'root$-1'
        
        def __init__(self):
            pass 
        
        def __repr__(self) -> str:
            return f'PoSDependencyLabeler()'
        
        def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
            indexes, rels = ['' for _ in range(len(graph))], ['' for _ in range(len(graph))]
            for arc in graph.arcs:
                pos = graph[arc.HEAD].UPOS
                if arc.side == -1: # left arc 
                    index = sum(graph[i].UPOS == pos for i in range(arc.DEP+1, arc.HEAD+1))
                else:
                    index = -sum(graph[i].UPOS == pos for i in range(arc.HEAD, arc.DEP))
                indexes[arc.DEP-1] = f'{pos}{self.SEP}{index}'
                rels[arc.DEP-1] = arc.REL 
            return indexes, rels 
        
        def decode(self, indexes: List[str], rels: List[str], tags: List[str]) -> List[Arc]:
            n, arcs, tags = len(indexes), [], ['<bos>'] + tags
            for idep, (label, rel) in enumerate(zip(indexes, rels)):
                dep = idep + 1
                pos, index = label.split(self.SEP)
                index = int(index)
                if index > 0: # head is at right 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(dep+1, n+1)]
                    index -= 1
                else: # head is at left 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(0, dep)]
                arcs.append(Arc(candidates[index], dep, rel))
            return arcs 
        
        def decode_postprocess(self, indexes: List[str], rels: List[str], tags: List[str]) -> Tuple[List[Arc], bool]:
            n, arcs, tags = len(indexes), [], ['<bos>'] + tags
            adjacent = torch.zeros(n+1, n+1, dtype=torch.bool)
            well_formed = True 
            for idep, (label, rel) in enumerate(zip(indexes, rels)):
                dep = idep + 1
                pos, index = label.split(self.SEP)
                index = int(index)
                if index > 0: # head is at right 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(dep+1, n+1)]
                    index -= 1
                    index = min(index, len(candidates)-1)
                else: # head is at left 
                    candidates = [i for i, tag in enumerate(tags) if tag == pos and i in range(0, dep)]
                    index = max(-len(candidates), index)
                if (len(candidates) == 0) or forms_cycles(adjacent, dep, candidates[index]) or (adjacent[:, 0].any().item() and candidates[index] == 0):
                    well_formed = False
                    if not adjacent[:, 0].any().item():
                        head = 0
                    else:
                        head = [h for h in set(range(1,n+1)) if not forms_cycles(adjacent, dep, h) and h != dep].pop(0)
                else:
                    head = candidates[index]
                arcs.append(Arc(head, dep, rel if head != 0 else 'root'))
                adjacent[dep, head] = True 
            return arcs, well_formed

        def test(self, graph: CoNLL.Graph) -> bool:
            indexes, rels = self.encode(graph)
            rec1 = graph.rebuild(self.decode(indexes, rels, graph.UPOS))
            rec2, well_formed = self.decode_postprocess(indexes, rels, graph.UPOS)
            rec2 = graph.rebuild(rec2)
            return graph == rec1 == rec2 and well_formed 
            
    def __init__(
        self,
        model: PoSDependencyModel, 
        input_tkzs: List[AbstractTokenizer],
        target_tkzs: List[AbstractTokenizer],
        gold: bool,
        device: str
    ):
        super().__init__(model, input_tkzs, target_tkzs, device)
        self.labeler = self.Labeler()
        self.TRANSFORM_ARGS = [input_tkzs, *target_tkzs, self.labeler]
        self.gold = gold
        
    @classmethod
    def transform(
        cls,
        graph: CoNLL.Graph,
        input_tkzs: List[Tokenizer],
        INDEX: Tokenizer,
        REL: Tokenizer,
        TAG: Tokenizer,
        labeler: PoSDependencyParser.Labeler
    ):
        if not graph._transformed:
            for tkz in input_tkzs:
                graph.__setattr__(tkz.name, tkz.encode(getattr(graph, tkz.field)).pin_memory())
            indexes, rels = labeler.encode(graph)
            graph.TAG = TAG.encode(graph.UPOS).pin_memory()
            graph.INDEX = INDEX.encode(indexes).pin_memory()
            graph.REL = REL.encode(rels).pin_memory()
            graph._transformed = True 
            
    def collate(self, graphs: List[CoNLL.Graph]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[CoNLL.Graph]]:
        inputs = [tkz.batch([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.input_tkzs]
        targets = [torch.cat([getattr(graph, tkz.name) for graph in graphs]) for tkz in self.target_tkzs]
        masks = [create_mask(list(map(len, graphs)))]
        return inputs, targets, masks, graphs
    
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = Parser.add_arguments(argparser)
        argparser.add_argument('-gold', '--gold', action='store_true', help='Whether to use gold PoS-tags')
        return argparser 
    
    
    def _pred(
        self,
        graph: CoNLL.Graph, 
        index_pred: torch.Tensor, 
        rel_pred: torch.Tensor,
        tag_pred: torch.Tensor
    ) -> Tuple[CoNLL.Graph, bool]:
        """Performs dependency graph reconstruction.

        Args:
            graph (CoNLL.Graph): Input dependency graph.
            index_pred (torch.Tensor ~ seq_len): Index prediction.
            rel_pred (torch.Tensor ~ seq_len): Relation prediction.
            tag_pred (torch.Tensor ~ seq_len): Tag prediction.

        Returns:
            Tuple[CoNLL.Graph, bool]: Predicted dependency graph and whether the sequence of 
                components conforms a well-formed dependency graph.
        """
        tags = graph.UPOS if self.gold else self.TAG.decode(tag_pred)
        rec, well_formed = self.labeler.decode_postprocess(self.INDEX.decode(index_pred), self.REL.decode(rel_pred), tags)
        return graph.rebuild(rec), well_formed
    
    def train_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Loss and control metrics.
        """
        indexes, rels, tags = targets
        s_index, s_rel, s_tag = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(s_index, s_rel, s_tag, indexes, rels, tags)
        control = dict(INDEX=acc(s_index, indexes), REL=acc(s_rel, rels))
        if not self.gold:
            control |= dict(TAG=acc(s_tag, tags))
        return loss, control
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Iterator[CoNLL.Graph]:
        """Prediction step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[CoNLL.Graph]): List of input dependency graphs.

        Returns:
            Iterator[CoNLL.Graph]: Predicted dependency graph.
        """
        lens = masks[0].sum(-1).tolist()
        index_preds, rel_preds, tag_preds = self.model.predict(inputs[0], inputs[1:], *masks)
        preds, _ = zip(*map(self._pred, graphs, index_preds.split(lens), rel_preds.split(lens), tag_preds.split(lens)))
        return preds  
    
    @torch.no_grad()
    def control_step(
        self, 
        inputs: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        graphs: List[CoNLL.Graph]
    ) -> Tuple[Dict[str, float], Iterator[CoNLL.Graph]]:
        """Control (prediction + evaluation) step.

        Args:
            inputs (List[torch.Tensor]): List of batched and padded inputs.
            targets (List[torch.Tensor]): List of batched and concatenated targets.
            masks (List[torch.Tensor]): List of padding masks.
            graphs (List[CoNLL.Graph]): List of input dependency graphs.

        Returns:
            Tuple[Dict[str, float], Iterator[CoNLL.Graph]]: Control metrics and predicted dependency graphs.
        """
        indexes, rels, tags, mask = *targets, *masks
        lens = mask.sum(-1).tolist()
        loss, index_preds, rel_preds, tag_preds = self.model.control(inputs[0], inputs[1:], *targets, mask)
        preds, well_formed = zip(*map(self._pred, graphs, index_preds.split(lens), rel_preds.split(lens), tag_preds.split(lens)))
        control = dict(INDEX=acc(index_preds, indexes), REL=acc(rel_preds, rels), loss=loss.item(), well_formed=avg(well_formed)*100)
        if not self.gold:
            control |= dict(TAG=acc(tag_preds, tags))
        return control, preds
    
            
    @classmethod 
    def build(
        cls, 
        data: Union[CoNLL, str],
        enc_conf: Config,
        word_conf: Config, 
        char_conf: Optional[Config] = None,
        gold: bool = False,
        pretest: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 1,
        **_
    ) -> PoSDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data, num_workers)
        
        if word_conf.pretrained:
            input_tkzs = [PretrainedTokenizer('WORD', 'FORM', word_conf.pretrained)]
            in_confs = [word_conf | input_tkzs[-1].conf, None]
        else:
            input_tkzs = [Tokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            # train input tokenizers
            for tkz in input_tkzs:
                tkz.train(*flatten(getattr(graph, tkz.field) for graph in data))
                
            # update configurations
            for conf, tkz in zip(in_confs, input_tkzs):
                conf.update(**tkz.conf())
            
        # train target tokenizers 
        index_tkz = Tokenizer('INDEX')
        rel_tkz = Tokenizer('REL')
        tag_tkz = Tokenizer('TAG', 'UPOS')
        labeler = cls.Labeler()
        if pretest:
            assert all(parallel(labeler.test, data, num_workers=num_workers, name=f'{cls.NAME}[pretest]'))
        indexes, rels = map(flatten, zip(*parallel(labeler.encode, data, num_workers=num_workers, name=f'{cls.NAME}[encode]')))
        index_tkz.train(*indexes)
        rel_tkz.train(*rels)
        tag_tkz.train(*flatten(graph.UPOS for graph in data))

        rel_conf = rel_tkz.conf 
        rel_conf.special_indices.append(rel_tkz.vocab['root'])
        model = cls.MODEL(enc_conf, *in_confs, index_tkz.conf, rel_conf, tag_tkz.conf if not gold else None).to(device)
        return cls(model, input_tkzs, [index_tkz, rel_tkz, tag_tkz], gold, device)
        
            