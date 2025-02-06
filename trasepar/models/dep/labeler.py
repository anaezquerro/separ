
from typing import List, Tuple 

from trasepar.structs import Arc 
from trasepar.data import CoNLL 

class DependencyLabeler:
    """Shared methods of a dependency labeler."""
    
    def preprocess(self, graph: CoNLL.Graph) -> List[List[Arc]]:
        return list(graph.planes.values())
    
    def encode(self, graph: CoNLL.Graph) -> Tuple[List[str], List[str]]:
        raise NotImplementedError
    
    def decode(self, labels: List[str], rels: List[str]) -> List[Arc]:
        raise NotImplementedError
    
    def decode_postprocess(self, labels: List[str], rels: List[str]) -> Tuple[List[Arc], bool]:
        raise NotImplementedError
    
    def test(self, graph: CoNLL.Graph) -> bool:
        raise NotImplementedError
    
    def theoretical(self, graph: CoNLL.Graph) -> CoNLL.Graph:
        """Returns the resulting graph after and the theoretical encoding -> decoding process.

        Args:
            graph (CoNLL.Graph): Input dependency metric.

        Returns:
            CoNLL.Graph: Output dependency graph.
        """
        labels, rels = self.encode(graph)
        recovered = graph.rebuild(self.decode_postprocess(labels, rels)[0])
        return recovered 
    
    def empirical(self, graph: CoNLL.Graph, known_labels: List[str], known_rels: List[str], LABEL: str, REL: str) -> CoNLL.Graph:
        """Returns the resulting graph after an empirical encoding -> decoding process.

        Args:
            graph (CoNLL.Graph): Input dependency graph.
            known (Set[str]): Set of known labels (the encoding is only allowed to use these labels).
            known_rels (List[str]): Set of known relations.
            REL (str): Default dependency relation for those arcs that are generated.

        Returns:
            CoNLL.Graph: Output dependency graph.
        """
        labels, rels = self.encode(graph)
        labels = [label if label in known_labels else LABEL for label in labels]
        rels = [rel if rel in known_rels else REL for rel in rels]
        rec1, _ = self.decode_postprocess(labels, rels)
        return graph.rebuild(rec1)