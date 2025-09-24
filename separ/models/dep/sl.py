from __future__ import annotations
import torch

from separ.data import CoNLL, Arc, forms_cycles, candidates_no_cycles, TargetTokenizer
from separ.utils import ControlMetric, acc, avg, DependencyMetric, TaggingMetric, Config, flatten, bar
from separ.models.tag import Tagger
from separ.models.dep.parser import DependencyParser
from separ.models.tag.model import TagModel

class DependencySLParser(DependencyParser, Tagger):
    MODEL = TagModel
    
    class Labeler:
        PARAMS = set()
        """Shared methods of a dependency labeler."""
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str]]:
            raise NotImplementedError
        
        def decode(self, labels: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            raise NotImplementedError
        
        def test(self, tree: CoNLL.Tree) -> bool:
            rec, well_formed = self.decode(*self.encode(tree))
            return tree.rebuild_from_arcs(rec) == tree and well_formed
        
        def theoretical(self, tree: CoNLL.Tree) -> CoNLL.Tree:
            """Returns the resulting tree after and the theoretical encoding -> decoding process.

            Args:
                tree (CoNLL.Tree): Input dependency metric.

            Returns:
                CoNLL.Tree: Output dependency tree.
            """
            labels, rels = self.encode(tree)
            recovered = tree.rebuild_from_arcs(self.decode(labels, rels)[0])
            return recovered 
        
        def empirical(self, tree: CoNLL.Tree, known_labels: list[str], known_rels: list[str], LABEL: str, REL: str) -> CoNLL.Tree:
            """Returns the resulting tree after an empirical encoding -> decoding process.

            Args:
                tree (CoNLL.Tree): Input dependency tree.
                known (Set[str]): Set of known labels (the encoding is only allowed to use these labels).
                known_rels (list[str]): Set of known relations.
                REL (str): Default dependency relation for those arcs that are generated.

            Returns:
                CoNLL.Tree: Output dependency tree.
            """
            labels, rels = self.encode(tree)
            labels = [label if label in known_labels else LABEL for label in labels]
            rels = [rel if rel in known_rels else REL for rel in rels]
            rec1, _ = self.decode(labels, rels)
            return tree.rebuild_from_arcs(rec1)
        
        def is_valid(self, adjacent: torch.Tensor, dep: int, head: int) -> bool:
            return not (forms_cycles(adjacent, dep, head) or adjacent[dep].any() or dep == 0 or (head == 0 and adjacent[:, head].any()))
        
        def postprocess(self, adjacent: torch.Tensor, rels: list[str]) -> list[Arc]:
            adjacent = adjacent.clone()
            arcs = [Arc(head, dep, rels[dep-1]) for dep, head in adjacent.nonzero().tolist()]
            no_assigned = sorted(set((adjacent.sum(-1) == 0).nonzero().flatten().tolist()[1:]))
            for dep in no_assigned:
                heads = candidates_no_cycles(adjacent, dep)
                head = heads.pop(0)
                arcs.append(Arc(head, dep, rels[dep-1]))
                adjacent[dep, head] = True
            return sorted(arcs)
        
    def __init__(self, **_):
        super().__init__(**_)
        self.lab = self.Labeler(**_)
        
    @property
    def PARAMS(cls):
        return cls.Labeler.PARAMS
        
    @property
    def TAG_METRIC(self) -> TaggingMetric:
        return TaggingMetric(self.target_tkzs)
    
    def _pred(self, tree: CoNLL.Tree, *preds: list[torch.Tensor]) -> tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(*[tkz.decode(pred) for tkz, pred in zip(self.target_tkzs, preds)])
        return tree.rebuild_from_arcs(rec), well_formed
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            for name, labels in zip(self.lab.NAMES, self.lab.encode(tree)):
                if name not in tree.FIELDS:
                    setattr(tree, name, labels)
            tree.transformed = True 
        return tree 
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> list[CoNLL.Tree]:
        preds, _ = zip(*super().pred_step(inputs, masks, trees))
        return preds  
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, TaggingMetric, DependencyMetric, list[CoNLL.Tree]]:
        scores = self.model(inputs[0], inputs[1:], *masks)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.TAG_METRIC(preds, targets), self.METRIC(pred_trees, trees), pred_trees
    
    @torch.no_grad()
    def debug_step(
        self, 
        masks: list[torch.Tensor], 
        targets: list[torch.Tensor], 
        trees: list[CoNLL.Tree]
    ) -> tuple[ControlMetric, TaggingMetric, DependencyMetric, list[CoNLL.Tree]]:
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(targets, masks)]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, targets, targets))), well_formed=avg(well_formed)*100)
        return control, self.TAG_METRIC(targets, targets), self.METRIC(pred_trees, trees), pred_trees

    @classmethod
    def build_targets(
        cls,
        lab: DependencySLParser.Labeler, 
        data: CoNLL
    ) -> list[TargetTokenizer]:
        target_tkzs = [TargetTokenizer(name) for name in lab.NAMES]
        labels = zip(*bar(map(lab.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]'))
        for tkz, _labels in zip(target_tkzs, map(flatten, labels)):
            tkz.train(_labels)
        return target_tkzs

    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        **_
    ) -> DependencySLParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        input_tkzs, in_confs = cls.build_inputs(data, word_conf, tag_conf, char_conf)
            
        target_tkzs = cls.build_targets(cls.Labeler(**_), data)
        
        return cls(
            input_tkzs=input_tkzs, 
            target_tkzs=target_tkzs,
            model_confs=[enc_conf, *in_confs] + [tkz.conf for tkz in target_tkzs], 
            **_
        )
