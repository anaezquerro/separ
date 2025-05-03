from typing import Tuple, List, Iterator 
import torch

from separ.data import CoNLL, Arc, forms_cycles, candidates_no_cycles
from separ.utils import ControlMetric, acc, avg, DependencyMetric
from separ.models.tag import Tagger
from separ.models.tag.model import TagModel

class DependencySLParser(Tagger):
    MODEL = TagModel
    DATASET = [CoNLL]
    PARAMS = []
    
    class Labeler:
        """Shared methods of a dependency labeler."""
        
        def encode(self, tree: CoNLL.Tree) -> Tuple[List[str], List[str]]:
            raise NotImplementedError
        
        def decode(self, labels: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
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
            recovered = tree.rebuild(self.decode(labels, rels)[0])
            return recovered 
        
        def empirical(self, tree: CoNLL.Tree, known_labels: List[str], known_rels: List[str], LABEL: str, REL: str) -> CoNLL.Tree:
            """Returns the resulting tree after an empirical encoding -> decoding process.

            Args:
                tree (CoNLL.Tree): Input dependency tree.
                known (Set[str]): Set of known labels (the encoding is only allowed to use these labels).
                known_rels (List[str]): Set of known relations.
                REL (str): Default dependency relation for those arcs that are generated.

            Returns:
                CoNLL.Tree: Output dependency tree.
            """
            labels, rels = self.encode(tree)
            labels = [label if label in known_labels else LABEL for label in labels]
            rels = [rel if rel in known_rels else REL for rel in rels]
            rec1, _ = self.decode(labels, rels)
            return tree.rebuild(rec1)
        
        def is_valid(self, adjacent: torch.Tensor, dep: int, head: int) -> bool:
            return not (forms_cycles(adjacent, dep, head) or adjacent[dep].any() or dep == 0 or (head == 0 and adjacent[:, head].any()))
        
        def postprocess(self, adjacent: torch.Tensor, rels: List[str]) -> List[Arc]:
            adjacent = adjacent.clone()
            arcs = [Arc(head, dep, rels[dep-1] if head != 0 else 'root') for dep, head in adjacent.nonzero().tolist()]
            no_assigned = sorted(set((adjacent.sum(-1) == 0).nonzero().flatten().tolist()[1:]))
            for dep in no_assigned:
                heads = candidates_no_cycles(adjacent, dep)
                head = heads.pop(0)
                arcs.append(Arc(head, dep, rels[dep-1] if head != 0 else 'root'))
                adjacent[dep, head] = True
            return sorted(arcs)
    
    @property 
    def METRIC(self) -> DependencyMetric:
        return DependencyMetric()
    
    def _pred(self, tree: CoNLL.Tree, *preds: List[torch.Tensor]) -> Tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(*[tkz.decode(pred) for tkz, pred in zip(self.target_tkzs, preds)])
        return tree.rebuild_from_arcs(rec), well_formed
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        trees: List[CoNLL.Tree]
    ) -> Iterator[CoNLL.Tree]:
        preds, _ = zip(*super().pred_step(self.model, inputs, masks, trees))
        return preds  
    
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor], 
        targets: List[torch.Tensor], 
        trees: List[CoNLL.Tree]
    ) -> Tuple[ControlMetric, DependencyMetric]:
        words, *feats, mask = *inputs, *masks
        lens = mask.sum(-1).tolist()
        scores = self.model(words, feats, mask)
        loss = self.model.loss(scores, targets)
        preds = self.model.predict(scores)
        pred_trees, well_formed = zip(*map(self._pred, trees, *[pred.split(lens) for pred in preds]))
        control = ControlMetric(**dict(zip(self.TARGET_FIELDS, map(acc, preds, targets))), loss=loss.detach(), well_formed=avg(well_formed)*100)
        return control, self.METRIC(pred_trees, trees)