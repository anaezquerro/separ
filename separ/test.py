from typing import List 
from separ.models import * 
from separ.data import CoNLL, PTB
from separ.utils import bar

CON_TEST = [
    IndexConstituencyParser.Labeler(rel=False),
    IndexConstituencyParser.Labeler(rel=True),
    TetraTaggingConstituencyParser.Labeler(),
]

DEP_TEST = [
    IndexDependencyParser.Labeler(rel=False),
    IndexDependencyParser.Labeler(rel=True),
    PoSDependencyParser.Labeler(),
    BracketDependencyParser.Labeler(k=1),
    BracketDependencyParser.Labeler(k=2),
    ArcEagerDependencyParser.System(1, 1, None),
    HierarchicalBracketDependencyParser.Labeler(variant='proj'),
    HierarchicalBracketDependencyParser.Labeler(variant='nonp'),
    Bit4DependencyParser.Labeler(),
    Bit7DependencyParser.Labeler(),
    HexaTaggingDependencyParser.Labeler(proj='head')
    
]

SDP_TEST = [
    IndexSemanticParser.Labeler(rel=False),
    IndexSemanticParser.Labeler(rel=True),
    BracketSemanticParser.Labeler(k=3),
    Bit4kSemanticParser.Labeler(k=3),
    Bit6kSemanticParser.Labeler(k=3),
    HierarchicalBracketSemanticParser.Labeler()
]

