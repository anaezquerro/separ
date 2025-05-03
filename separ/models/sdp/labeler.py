
from typing import List, Tuple, Set
import torch 

from separ.data import SDP 
from separ.utils import flatten
from separ.structs import Arc, adjacent_from_arcs

class SemanticLabeler:
    """Shared methods of a semantic labeler."""
    SEP = '$'
    DEFAULT_REL = 'punct'

    def __repr__(self) -> str:
        raise NotImplementedError 
    
    def preprocess(self, graph: SDP.Graph) -> List[List[Arc]]:
        return list(graph.planes.values())
    
    def encode(self, graph: SDP.Graph) -> List[str]:
        raise NotImplementedError
    
    def decode(self, labels: List[str]) -> List[Arc]:
        raise NotImplementedError
    
    def decode_postprocess(self, labels: List[str]) -> Tuple[List[Arc], torch.Tensor, bool]:
        raise NotImplementedError
    
    def test(self, graph: SDP.Graph) -> bool:
        """Tests the encoding, decoding and post-processing steps.

        Args:
            graph (SDP.Graph): Input graph.

        Returns:
            bool: Whether the labeler implementation is correct.
        """
        adjacent = adjacent_from_arcs(flatten(self.preprocess(graph)), len(graph))
        labels = self.encode(graph)
        rec1 = adjacent_from_arcs(self.decode(labels), len(graph))
        _, rec2, well_formed = self.decode_postprocess(labels)
        return well_formed and bool(((adjacent == rec1) & (adjacent == rec2)).all())
    
    def theoretical(self, graph: SDP.Graph) -> SDP.Graph:
        """Returns the resulting graph after and the theoretical encoding -> decoding process.

        Args:
            graph (SDP.Graph): Input semantic graph.

        Returns:
            SDP.Graph: Output semantic graph.
        """
        return graph.rebuild(flatten(self.preprocess(graph)))
    
    def empirical(self, graph: SDP.Graph, known: Set[str], REL: str) -> SDP.Graph:
        """Returns the resulting graph after an empirical encoding -> decoding process.

        Args:
            graph (SDP.Graph): Input semantic graph.
            known (Set[str]): Set of known labels (the encoding is only allowed to use these labels).
            REL (str): Default dependency relation for those arcs that are generated.

        Returns:
            SDP.Graph: Output semantic graph.
        """
        labels = [label if label in known else self.DEFAULT for label in self.encode(graph)]
        rec, _, _ = self.decode_postprocess(labels)
        # use the gold dependency relations or the default 
        for arc in rec:
            if graph.ADJACENT[arc.DEP, arc.HEAD]:
                arc.REL = graph.LABELED_ADJACENT[arc.DEP, arc.HEAD]
            else:
                arc.REL = REL 
        return graph.rebuild(rec)
    
    def complete_encode(self, graph: SDP.Graph) -> Tuple[List[str], List[str]]:
        labels = self.encode(graph)
        rels = [[] for _ in range(len(graph))]
        supported = sorted(flatten(self.preprocess(graph)))  # arcs that are encoded 
        for arc in supported: # arcs are already sorted
            rels[arc.DEP-1].append(arc.REL)
        return labels, [self.SEP.join(rel) for rel in rels]
    
    def complete_decode(self, labels: List[str], rels: List[str]) -> List[Arc]:
        recovered = self.decode(labels)
        rels = [rel.split(self.SEP) for rel in rels]
        for arc in recovered: # arcs are already sorted
            arc.REL = rels[arc.DEP-1].pop(0)
        return recovered 
    
    def complete_decode_postprocess(self, labels: List[str], rels: List[str]) -> Tuple[List[Arc], bool]: 
        recovered, _, well_formed = self.decode_postprocess(labels)
        rels = [rel.split(self.SEP) for rel in rels]
        for arc in recovered: # arcs are already sorted
            if len(rels[arc.DEP-1]) > 0:
                arc.REL = rels[arc.DEP-1].pop(0)
            else:
                arc.REL = self.DEFAULT_REL
                well_formed = False
        return recovered, well_formed