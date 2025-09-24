from __future__ import annotations
from argparse import ArgumentParser

from separ.models.con.parser import ConstituencySLParser
from separ.utils import Config, flatten, bar
from separ.data import PTB, InputTokenizer, TargetTokenizer, CharacterTokenizer, PretrainedTokenizer

def extra_leaves(tag: str) -> str:
    if PTB.Tree.UNARY not in tag:
        return ''
    else:
        *extra, _ = tag.split(PTB.Tree.UNARY)
        return PTB.Tree.UNARY.join(extra)

class IndexConstituencyParser(ConstituencySLParser):
    NAME = 'con-idx'
    
    class Labeler:
        PARAMS = {'rel'}
        
        def __init__(self, rel: bool, ROOT: str = 'S', **_):
            self.rel = rel 
            self.ROOT = ROOT
            
        @property
        def NAME(self) -> str:
            return 'con-rel' if self.rel else 'con-abs'
            
        @property
        def NAMES(self) -> list[str]:
            if self.rel:
                return ['INDEX(R)', 'CON', 'LEAF']
            else:
                return ['INDEX(A)', 'CON', 'LEAF']
            
        def __repr__(self) -> str:
            return f'IndexConstituencyLabeler(rel={self.rel})'
            
        def encode(self, tree: PTB.Tree) -> tuple[list[str], list[str], list[str]]:
            collapsed = tree.collapse_unary()
            indexes = [0 for _ in range(len(tree))]
            lens = [len(tree)+1 for _ in range(len(tree))]
            cons = ['' for _ in range(len(tree))]
            for span in sorted(collapsed.spans, key=lambda s: (s.LEFT, len(s))):
                for i in range(span.LEFT, span.RIGHT-1):
                    indexes[i] += 1 
                    if lens[i] > len(span):
                        lens[i] = len(span)
                        cons[i] = span.LABEL
            leaves = list(map(extra_leaves, collapsed.POS))
            if self.rel:
                indexes = self.relativize(indexes)
            return list(map(str, indexes)), cons, leaves
    
        def relativize(self, indexes: list[int | str]) -> list[int]:
            return [int(indexes[0])] + [int(indexes[i]) - int(indexes[i-1]) for i in range(1, len(indexes))]
             
        def decode(self, indexes: list[str], cons: list[str], leaves: list[str]) -> tuple[list[PTB.Span], bool]:
            stack, spans = [], []
            indexes = self.relativize(indexes) if not self.rel else list(map(int, indexes))
            for i, (index, con) in enumerate(zip(indexes, cons)):
                if index > 0: # open new spans (LEFT, RIGHT, LABEL)
                    stack += [(i, None, None) for _ in range(index-1)]
                    stack.append((i, None, con))
                elif index < 0 and len(stack) > 0: # close previous spans 
                    cnt, p = 0, len(stack)-1
                    while cnt < abs(index) and p >= 0:
                        if len(stack) > 0 and stack[p][-1] is not None:
                            cnt += 1 
                            span = stack.pop(p)
                            spans.append(PTB.Span(span[0], i+1, span[-1]))
                        p -= 1
                    p = len(stack) - 1
                    while p >= 0:
                        if stack[p][-1] is None or stack[p][-1] == con:
                            stack[p] = (stack[p][0], None, con)
                            break 
                        p -= 1
            spans = sorted(spans + [PTB.Span(i, i+1, tag) for i, tag in enumerate(leaves) if tag != ''])
            return spans, all(span.LABEL is not None for span in spans) and len(stack) == 0
        
        def test(self, tree: PTB.Tree) -> bool:
            spans, well_formed = self.decode(*self.encode(tree))
            rec = PTB.Tree.from_spans(tree.preterminals, spans).recover_unary()
            return rec == tree and well_formed
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = ConstituencySLParser.add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser
        
            