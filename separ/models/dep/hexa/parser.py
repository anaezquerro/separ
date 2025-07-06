from __future__ import annotations
from argparse import ArgumentParser
import torch 

from separ.models.dep.parser import DependencySLParser
from separ.data import CoNLL, InputTokenizer, TargetTokenizer, PretrainedTokenizer, CharacterTokenizer, PTB, Arc
from separ.utils import Config, flatten, bar

class HexaTaggingDependencyParser(DependencySLParser):
    """HexaTagging dependency parser from [Amini et al. (2023)](https://aclanthology.org/2023.acl-short.124/)."""
    NAME = 'dep-hexa'
    PARAMS = ['proj']
    
    class Labeler(DependencySLParser.Labeler):
        LEFT = 'L'
        RIGHT = 'R'
        ROOT = '$'
        
        def __init__(self, proj: str):
            self.proj = proj
        
        def __repr__(self) -> str:
            return f'HexaTaggingDependencyLabeler(proj={self.proj})'
        
        def encode(self, tree: CoNLL.Tree) -> tuple[list[str], list[str], list[str]]:
            tree = tree.projectivize(self.proj)
            if len(tree) == 1:
                return [], [], tree.DEPREL

            # Step 1: Transform dependency tree into BHT 
            stack = []
            self.dep2tree(0, tree, stack)
            ctree = stack[0][1]
            
            # Step 2: Obtain labels of tree
            hexas, fences, cons = self.encode_bht(ctree, [], ['>'])
            fences = list(map(''.join, zip(fences, cons)))
            return hexas[1:-1], fences, tree.DEPREL
        
        def decode(self, hexas: list[str], fences: list[str], rels: list[str]) -> tuple[list[Arc], bool]:
            if len(rels) <= 1:
                return [Arc(0, 1, 'root')], True 
            # Step 1: Obtain the BHT 
            tree, well_formed = self.decode_bht(['>'] + hexas + ['<'], *zip(*map(list, fences)), len(rels))

            # Step 2: Recover the dependency tree
            arcs = []
            btree = tree.rebuild_terminals(list(map(str, range(1, len(rels)+1))))
            self.tree2dep(PTB.Tree(self.ROOT, [btree]), arcs)
            for arc, rel in zip(sorted(arcs), rels):
                arc.REL = rel 
            return arcs, well_formed

        def dep2tree(self, head: int, tree: CoNLL.Tree, stack: list[int]):
            stack.append(head)
            # get left and right dependencies
            lefts, rights = [], []
            for arc in tree.arcs:
                if arc.HEAD == head:
                    if arc.DEP < arc.HEAD:
                        lefts.append(arc.DEP)
                    else:
                        rights.append(arc.DEP)
            for dep in sorted(lefts, reverse=True):
                self.dep2tree(dep, tree, stack)
                left = stack.pop(-1)
                right = stack.pop(-1)
                if isinstance(left, int):
                    left = PTB.Tree.from_leaf(str(left), tree.FORM[left-1])
                if isinstance(right, int):
                    right = PTB.Tree.from_leaf(str(right), tree.FORM[right-1])
                stack.append(PTB.Tree(self.RIGHT, deps=[left, right]))
            for dep in rights:
                self.dep2tree(dep, tree, stack)
                right = stack.pop(-1)
                left = stack.pop(-1)
                if isinstance(left, int):
                    left = PTB.Tree.from_leaf(str(left), tree.FORM[left-1])
                if isinstance(right, int):
                    right = PTB.Tree.from_leaf(str(right), tree.FORM[right-1])
                stack.append(PTB.Tree(self.LEFT, deps=[left, right]))
                
        def tree2dep(self, node: PTB.Tree, arcs: list[Arc]):
            if node.is_preterminal(): 
                return node.deps[0]
            elif node.is_terminal():
                return node
            left = self.tree2dep(node.deps[0], arcs)
            if len(node.deps) == 1:
                arcs.append(Arc(0, int(left.label), None))
                return 
            else:
                right = self.tree2dep(node.deps[1], arcs)
            if node.label[0] == self.LEFT:
                arcs.append(Arc(int(left.label), int(right.label), None))
                return left
            arcs.append(Arc(int(right.label), int(left.label), None))
            return right
        
        def encode_bht(self, tree: PTB.Tree, tags: list[str], fences: list[str]) -> list[str]:
            if len(tree.deps) == 1:
                return ['>'], ['>'], [tree.label]
            left, right = tree.deps 
            if left.is_preterminal():
                left_tags, left_fences, left_cons = ['>'], [],  []
            else:
                left_tags, left_fences, left_cons = self.encode_bht(left, [], ['>'])
            if right.is_preterminal():
                right_tags, right_fences, right_cons = ['<'], [], []
            else:
                right_tags, right_fences, right_cons = self.encode_bht(right, [], ['<'])
            return left_tags + tags + right_tags, left_fences + fences + right_fences, left_cons + [tree.label] + right_cons

        
        def decode_bht(
            self, 
            tags: list[str],
            fences: list[str],
            cons: list[str], 
            n: int
        ) -> tuple[PTB.Tree, bool]:
            """Performs the decoding algorithm to transform a sequence of 
            hexatags into  BHT.
            
            Args:
                tags (list[str] ~ n): Sequence of token tags (>, <).
                fences (list[str] ~ (n-1)): Sequence of fencepost tags (>, <).
                cons (list[str] ~ (n-1)): Sequence of fencepost tags for constituents (L, R).
            
            Returns:
                tuple[PTB.Tree, bool]: Recovered BHT and whether the sequence of 
                tags is well-formed.
            """
            pos = list(map(str, range(1, n+1)))
            stack, well_formed = [], True
            for tag, fence, con in zip(tags, fences, cons):
                if (tag == '>' or len(stack) < 1) and (sum(s.label == self.ROOT for s in stack) < len(pos)):
                    stack.append(PTB.Tree(self.ROOT, deps=[PTB.Tree.from_leaf(pos[0], pos.pop(0))]))
                    well_formed &= (tag == '>')
                else:
                    stack[-1].deps.append(PTB.Tree.from_leaf(pos[0], pos.pop(0)))
                    stack.pop(-1)
                    well_formed &= (tag == '<')
                    
                if (fence == '>' or len(stack) <= 1) and (sum(s.label == self.ROOT for s in stack) < len(pos) + 1):
                    stack[-1].deps = [PTB.Tree(con, deps=stack[-1].deps)]
                    stack.append(stack[-1].deps[0])
                    well_formed &= (fence == '>')
                else:
                    last = stack.pop(-1)
                    last.label = con 
                    stack[-1].deps.append(last)
                    stack[-1] = last 
                    well_formed &= (fence == '<')
            stack[-1].deps.append(PTB.Tree.from_leaf(pos[0], pos.pop(0)))
            return stack[0].deps[0], well_formed
        
        def test(self, tree: CoNLL.Tree) -> bool:
            return super().test(tree.projectivize(mode='head'))
        
        def theoretical(self, tree: CoNLL.Tree) -> CoNLL.Tree:
            rec, _ = self.decode(*self.encode(tree))
            rec = tree.rebuild_from_arcs(rec)
            if self.proj:
                rec = rec.deprojectivize(self.proj)
            return rec 
        
        def empirical(
            self, 
            tree: CoNLL.Tree,
            known_hexas: list[str], 
            known_fences: list[str], 
            known_rels: list[str], 
            HEXA: str, 
            FENCE: str,
            REL: str
        ):
            hexas, fences, rels = self.encode(tree)
            hexas = [hexa if hexa in known_hexas else HEXA for hexa in hexas]
            fences = [fence if fence in known_fences else FENCE for fence in fences]
            rels = [rel if rel in known_rels else REL for rel in rels]
            rec, _ = self.decode(hexas, fences, rels)
            rec = tree.rebuild_from_arcs(rec)
            if self.proj:
                rec = rec.deprojectivize(self.proj)
            return rec 

    
    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        proj: Optional[str],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
        self.lab = self.Labeler(proj)
        self.proj = proj 
        
    @classmethod 
    def add_arguments(cls, argparser: ArgumentParser):
        argparser = super(HexaTaggingDependencyParser, cls).add_arguments(argparser)
        argparser.add_argument('--proj', default='head', type=str, choices=['head', 'head+path', 'path'], help='Pseudo-projective mode')
        return argparser 
    
    def transform(self, tree: CoNLL.Tree) -> CoNLL.Tree:
        if not tree.transformed:
            tree.HEXA, tree.FENCE, tree.REL = self.lab.encode(tree)
            tree.transformed = True 
        return tree
    
    def collate(self, trees: list[CoNLL.Tree]):
        inputs, (hmask, fmask, rmask), targets, _ = super().collate(trees)
        lens = rmask.sum(-1).tolist()
        hmask[:, 0] = False 
        for i, l in enumerate(lens):
            hmask[i,l-1] = False
            fmask[i,l-1] = False 
        return inputs, [hmask, fmask, rmask], targets, trees 
        
    def _pred(self, tree: CoNLL.Tree, *preds: list[torch.Tensor]) -> tuple[CoNLL.Tree, bool]:
        rec, well_formed = self.lab.decode(*[tkz.decode(pred) for tkz, pred in zip(self.target_tkzs, preds)])
        pred = tree.rebuild_from_arcs(rec)
        if self.proj:
            pred = pred.deprojectivize(self.proj)
        return pred, well_formed
   
    @classmethod 
    def build(
        cls, 
        data: str | CoNLL,
        enc_conf: Config,
        word_conf: Config, 
        tag_conf: Config | None = None,
        char_conf: Config | None = None,
        proj: str | None = None,
        device: int = 0,
        **_
    ) -> HexaTaggingDependencyParser:
        if isinstance(data, str):
            data = CoNLL.from_file(data)
        
        if 'pretrained' in word_conf:
            input_tkzs = [PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')]
            in_confs = [word_conf | input_tkzs[-1].conf, None, None]
        else:
            input_tkzs = [InputTokenizer('WORD', 'FORM')]
            in_confs = [word_conf]
            if tag_conf is not None:
                input_tkzs.append(InputTokenizer('TAG', 'UPOS'))
                in_confs.append(tag_conf)
            else:
                in_confs.append(None)
            if char_conf is not None:
                input_tkzs.append(CharacterTokenizer('CHAR', 'FORM'))
                in_confs.append(char_conf)
            else:
                in_confs.append(None)
                
            for tkz in input_tkzs:
                tkz.train(data)

            for conf, tkz in zip([c for c in in_confs if c is not None], input_tkzs):
                conf.update(tkz.conf)
            
        hexa_tkz = TargetTokenizer('HEXA')
        fence_tkz = TargetTokenizer('FENCE')
        rel_tkz = TargetTokenizer('REL')
        labeler = cls.Labeler(proj)
        hexas, fences, rels = map(flatten, zip(*bar(map(labeler.encode, data), total=len(data), leave=False, desc=f'{cls.NAME}[encode]')))
        hexa_tkz.train(hexas)
        fence_tkz.train(fences)
        rel_tkz.train(rels)
        return cls(
            input_tkzs, [hexa_tkz, fence_tkz, rel_tkz], 
            [enc_conf, *in_confs, hexa_tkz.conf, fence_tkz.conf, rel_tkz.conf], 
            proj, device 
        )