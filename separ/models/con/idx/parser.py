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
    PARAMS = ['rel']
    
    class Labeler:
        def __init__(self, rel: bool, ROOT: str = 'S'):
            self.rel = rel 
            self.ROOT = ROOT
            
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
        
    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        rel: bool,
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.rel = rel 
        self.lab = self.Labeler(rel)
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = ConstituencySLParser.add_arguments(argparser)
        argparser.add_argument('-rel', '--rel', action='store_true', help='Relative indexing')
        return argparser
    
    def transform(self, tree: PTB.Tree) -> PTB.Tree:
        if not tree.transformed:
            tree.INDEX, tree.CON, tree.LEAF = self.lab.encode(tree)
            tree.transformed = True 
        return tree

    @classmethod
    def build(
        cls, 
        data: str | PTB,
        enc_conf: Config,
        word_conf: Config,
        char_conf: Config | None = None,
        rel: bool = False,
        device: int = 0,
        **_
    ) -> IndexConstituencyParser:
        if isinstance(data, str):
            data = PTB.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            word_tkz = InputTokenizer('WORD', 'FORM')
            word_tkz.train(data)
            word_conf.update(word_tkz.conf)
            input_tkzs = [word_tkz]
            
            if char_conf is not None:
                char_tkz = CharacterTokenizer('CHAR', 'FORM')
                char_tkz.train(data)
                char_conf.update(char_tkz.conf)
                input_tkzs.append(char_tkz)
                
            in_confs = [word_conf, None, char_conf]
                    
        # train target tokenizers 
        index_tkz, con_tkz, leaf_tkz = TargetTokenizer('INDEX'), TargetTokenizer('CON'), TargetTokenizer('LEAF')
        labeler = cls.Labeler(rel=rel)
        indexes, cons, leaves = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        index_tkz.train(indexes)
        con_tkz.train(cons)
        leaf_tkz.train(leaves)
        return cls(input_tkzs, [index_tkz, con_tkz, leaf_tkz], [enc_conf, *in_confs, index_tkz.conf, con_tkz.conf, leaf_tkz.conf], rel, device)
            
        
            