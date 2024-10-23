
from __future__ import annotations
from typing import List, Iterator, Optional, Tuple, Dict 
import numpy as np 
import torch, re 

from separ.structs import AbstractDataset
from separ.utils import parallel, flatten


def split_ptb(line: str) -> Iterator[str]:
    """Iterates over the elements of a bracketed tree representation.
    
    Args:
        line (str): Bracketed tree representation.
        
    Returns: Iterator[str]
        Elements of the tree.
        
    Examples: 
    >>> line = "(S (NP-SBJ (PRP It) ) (VP (VBZ has) (NP (NP (DT no) (NN bearing) ) (PP-DIR (IN on) (NP (NP (PRP$ our) (NN work) (NN force) ) (NP-TMP (NN today) ))))) (. .) )"
    >>> list(split_ptb(line))[:10]
    ['(', 'S', '(', 'NP-SBJ', '(', 'PRP', 'It', ')']
    """
    item = ''
    for char in line.strip():
        if char == '(' or char == ')':
            if len(item) > 0:
                yield item 
            item = ''
            yield char
        elif char == ' ':
            if len(item) > 0:
                yield item
                item = ''
        else:
            item += char 
            
class PTB(AbstractDataset):
    EXTENSION: str = '.ptb'
    SEP: str = '\n'
    HEADER: str = ''
    
    class Tree:
        """Instance of a tree as the label component and attached dependants."""
        UNARY = '@'
        FIELDS = ['FORM', 'POS']
        
        def __init__(self, label: str, deps: List[PTB.Tree] = []):
            self.label = label 
            self.deps = deps 
            self.ID = None 
            self._transformed = False
            assert isinstance(label, str), self.label
            assert all(isinstance(dep, PTB.Tree) for dep in self.deps), self.deps
            self.build()
            
        def build(self):
            """Builds the labeled and unlabeled span matrix.
            
            Examples
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.spans 
            [
                Span(LEFT=0, RIGHT=4, LABEL=S),
                Span(LEFT=0, RIGHT=2, LABEL=A),
                Span(LEFT=2, RIGHT=4, LABEL=B)
            ]
            >>> tree.LABELED_MATRIX
            array([
                ['_', '_', 'A', '_', 'S'],
                ['_', '_', '_', '_', '_'],
                ['_', '_', '_', '_', 'B'],
                ['_', '_', '_', '_', '_'],
                ['_', '_', '_', '_', '_']
            ], dtype='<U1')
            """
            self.MATRIX = torch.zeros(len(self) + 1, len(self) + 1, dtype=torch.bool)
            self.LABELED_MATRIX = np.full((len(self)+1, len(self)+1), '_')
            for span in self.spans:
                self.MATRIX[span.LEFT, span.RIGHT] = True 
                self.LABELED_MATRIX[span.LEFT, span.RIGHT] = span.LABEL
                
        def copy(self) -> PTB.Tree:
            if self.is_terminal():
                return PTB.Tree(self.label, self.deps)
            else:
                return PTB.Tree(self.label, [dep.copy() for dep in self.deps])

        def __getitem__(self, index: int) -> PTB.Tree:
            """Index the dependants of the tree.

            Args:
                index (int): Index of the dependant.

            Returns:
                PTB.Tree: Dependant tree.
            """
            return self.deps[index]
        
        def __len__(self) -> int:
            """Returns the number of terminal nodes that are included in the tree."""
            return len(self.leaves)
        
        def __repr__(self) -> str:
            """Returns the bracketed representation of the tree.
            
            Examples:
            >>> tree = PTB.Tree('NP', deps=[PTB.Tree('world')])
            >>> tree
            (NP world)
            """
            if self.is_terminal():
                return self.label 
            else:
                return f'({self.label} ' + ' '.join(repr(dep) for dep in self.deps) + ')'
            
        def __eq__(self, other: PTB.Tree) -> bool:
            """Returns whether two trees are equal.

            Returns:
                bool: Whether the input tree is the same.
            """
            if not isinstance(other, PTB.Tree):
                return False 
            elif self.is_terminal():
                return other.is_terminal() and self.label == other.label 
            else:
                return self.label == other.label and len(self.deps) == len(other.deps) and \
                    all(dep1 == dep2 for dep1, dep2 in zip(self.deps, other.deps))
                
        def format(self) -> str:
            """Returns the PTB representation of the tree.

            Examples:
            >>> tree = PTB.Tree('NP', deps=[PTB.Tree('world')])
            >>> tree
            (NP world)
            """
            return repr(self)
        
        def is_terminal(self) -> bool:
            """Returns whether the tree is terminal (if it has no dependants)."""
            return len(self.deps) == 0
        
        def is_preterminal(self) -> bool:
            """Returns whether the tree is pre-terminal (only has one dependant that is terminal)."""
            return len(self.deps) == 1 and self.deps[0].is_terminal() 
        
        def is_binary(self) -> bool:
            """Whether the tree is binary or not."""
            if all(dep.is_preterminal() for dep in self.deps):
                return len(self.deps) == 2
            else:
                if len(self.deps) != 2:
                    return False 
                else:
                    return all(dep.is_binary() for dep in self.deps)
                
        def has_unary(self) -> bool:
            """Whether the tree has unary chains."""
            if self.is_preterminal():
                return False 
            if len(self.deps) == 1:
                return True
            return any(dep.has_unary() for dep in self.deps)
            
        def collapse_unary(self) -> PTB.Tree:
            """Collapse unary chains and returns a new tree.

            Returns:
                PTB.Tree: New tree with no unary chains.
            """
            if self.is_preterminal():
                return self
            elif len(self.deps) == 1:
                tree = PTB.Tree(f'{self.label}{self.UNARY}{self.deps[0].label}', deps=self.deps[0].deps)
                return tree.collapse_unary()
            else:
                return PTB.Tree(self.label, deps=[dep.collapse_unary() for dep in self.deps])
            
        def recover_unary(self) -> PTB.Tree:
            """Recovers unary chains and returns the original tree.

            Returns:
                PTB.Tree: New tree with unary chains.
            """
            if self.is_terminal():
                return self 
            elif self.UNARY in self.label:
                parent, *deps = self.label.split(self.UNARY)
                tree = PTB.Tree(parent, deps=[PTB.Tree(self.UNARY.join(deps), deps=self.deps)])
                return tree.recover_unary()
            else:
                return PTB.Tree(self.label, deps=[dep.recover_unary() for dep in self.deps])
            
        def rebuild_tags(self, tags: List[str]) -> PTB.Tree:
            if self.is_preterminal():
                return PTB.Tree(label=tags.pop(0), deps=self.deps)
            else:
                return PTB.Tree(self.label, deps=[dep.rebuild_tags(tags) for dep in self.deps])
                
        @property
        def depth(self) -> int:
            if self.is_preterminal():
                return 1
            else:
                return 1 + max(dep.depth for dep in self.deps)
            
        @property 
        def leaves(self) -> List[PTB.Tree]:
            """Returns the leaves (terminal nodes) of the tree.
                
            Examples:
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.leaves 
            [w1, w2, w3, w4]
            """
            if len(self.deps) == 0:
                return [self]
            else:
                leaves = []
                for dep in self.deps:
                    leaves += dep.leaves 
                return leaves
            
        @property
        def preterminals(self) -> List[PTB.Tree]:
            """Returns the preterminal nodes of the tree.
            
            Examples:
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.preterminals
            [(t1 w1), (t2 w2), (t3 w3), (t4 w4)]
            """
            if self.is_preterminal():
                return [self]
            else:
                preterminals = []
                for dep in self.deps:
                    preterminals += dep.preterminals
                return preterminals
        
        @property
        def constituents(self) -> List[str]:
            """Returns the list of constituents (breadth-first search) of the tree.
            
            Examples:
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.constituents
            ['S', 'A', 'B']
            """
            if self.is_preterminal():
                return []
            else:
                constituents = [self.label]
                for dep in self.deps:
                    constituents += dep.constituents
                return constituents
            
        @property 
        def FORM(self) -> List[str]:
            """Returns the words of the terminal nodes of the tree.
            
            Examples:
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.FORM
            ['w1', 'w2', 'w3', 'w4']
            """
            return [leaf.label for leaf in self.leaves]
        
        @property 
        def POS(self) -> List[str]:
            """Returns the PoS-tags of the preterminal nodes of the tree.
            
            Examples:
            >>> tree = PTB.Tree.from_ptb("(S (A (t1 w1) (t2 w2)) (B (t3 w3) (t4 w4)))")
            >>> tree.POS
            ['t1', 't2', 't3', 't4']
            """
            return [pre.label for pre in self.preterminals]
        
        @property
        def tags(self) -> List[str]:
            return [pre.label for pre in self.preterminals]
                
        @property 
        def spans(self) -> List[PTB.Span]:
            """Returns the list of spans."""
            return self.__spans()[0]
        
        def __spans(self, start: int = 0) -> Tuple[List[PTB.Span], int]:
            if self.is_terminal() or self.is_preterminal():
                return [], start+1
            else:
                spans, end = [], start 
                for dep in self.deps:
                    _spans, end = dep.__spans(end)
                    spans += _spans 
                return [PTB.Span(start, end, self.label), *spans], end 
            
        @classmethod
        def from_spans(cls, nodes: List[PTB.Tree], spans: List[PTB.Span]) -> PTB.Tree:
            """Builds a tree from a list of spans.

            Args:
                nodes (List[PTB.Tree]): List of preterminal nodes.
                spans (List[PTB.Span]): List of spans.

            Returns:
                PTB.Tree: Tree instance.
                
            Examples:
            >>> line = "(S (NP-SBJ-1 (DT The) (NN decision)) (VP (VBD was) (VP (VBN announced) (NP (-NONE- *-1)) (SBAR-TMP (IN after) (S (NP-SBJ (NN trading)) (VP (VBD ended)))))) (. .))"
            >>> tree = PTB.Tree.from_ptb(line)
            >>> tree.spans
            [
                Span(LEFT=0, RIGHT=9, LABEL=S),
                Span(LEFT=0, RIGHT=2, LABEL=NP-SBJ-1),
                Span(LEFT=2, RIGHT=8, LABEL=VP),
                Span(LEFT=3, RIGHT=8, LABEL=VP),
                Span(LEFT=4, RIGHT=5, LABEL=NP),
                Span(LEFT=5, RIGHT=8, LABEL=SBAR-TMP),
                Span(LEFT=6, RIGHT=8, LABEL=S),
                Span(LEFT=6, RIGHT=7, LABEL=NP-SBJ),
                Span(LEFT=7, RIGHT=8, LABEL=VP)
            ]
            >>> tree.has_unary()
            False 
            >>> PTB.Tree.from_spans(tree.preterminals, tree.spans)
            (S (NP-SBJ-1 (DT The) (NN decision)) (VP (VBD was) (VP (VBN announced) (NP (-NONE- *-1)) (SBAR-TMP (IN after) (S (NP-SBJ (NN trading)) (VP (VBD ended)))))) (. .))
            """
            for span in sorted(spans):
                nodes[span.LEFT] = PTB.Tree(span.LABEL, deps=[nodes[i] for i in range(span.LEFT, span.RIGHT) if nodes[i] is not None])
                for i in range(span.LEFT+1, span.RIGHT):
                    nodes[i] = None 
            return nodes[0].recover_unary()
            
        @classmethod
        def from_ptb(cls, line: str) -> PTB.Tree:
            """Builds a tree from the PTB format.

            Args:
                line (str): Line of the PTB representation.

            Returns:
                PTB.Tree: Tree instance.
                
            Examples:
            >>> line = "(S (NP-SBJ (NP (JJ Influential) (NNS members)) (PP (IN of) (NP (DT the) (NNP House) (NNP Ways) (CC and) (NNP Means) (NNP Committee)))) (VP (VBD introduced) (NP (NP (NN legislation)) (SBAR (WHNP-1 (WDT that)) (S (NP-SBJ-3 (-NONE- *T*-1)) (VP (MD would) (VP (VB restrict) (SBAR (WHADVP-2 (WRB how)) (S (NP-SBJ (DT the) (JJ new) (NN savings-and-loan) (NN bailout) (NN agency)) (VP (MD can) (VP (VB raise) (NP (NN capital)) (ADVP-MNR (-NONE- *T*-2)))))) (, ,) (S-ADV (NP-SBJ (-NONE- *-3)) (VP (VBG creating) (NP (NP (DT another) (JJ potential) (NN obstacle)) (PP (TO to) (NP (NP (NP (DT the) (NN government) (POS 's)) (NN sale)) (PP (IN of) (NP (JJ sick) (NNS thrifts)))))))))))))) (. .))"
            >>> tree = PTB.Tree.from_ptb(line)
            >>> tree.spans[0]
            Span(LEFT=0, RIGHT=40, LABEL=S)
            >>> tree.leaves[:10]
            [Influential, members, of, the, House, Ways, and, Means, Committee, introduced]
            """
            stack = []
            for item in split_ptb(line):
                # a node is being closed 
                if item == ')': 
                    # pop the stack until an opening bracket is found 
                    items = [stack.pop(-1)]
                    while items[-1] != '(':
                        items.append(stack.pop(-1))
                    items = items[:-1][::-1]
                    if isinstance(items[0], str) and isinstance(items[-1], str): # we are in the terminal case 
                        stack.append(PTB.Tree(items[0], deps=[PTB.Tree(items[1])]))
                    elif isinstance(items[0], str):
                        tree = PTB.Tree(items[0], deps=items[1:])
                        stack.append(tree)
                    else:
                        stack.append(items[0])
                else:
                    stack.append(item)
            assert len(stack) == 1, f'\n{len(stack)}\n' + '\n'.join(map(repr, stack))
            tree = stack[0]
            assert line == tree.format(), f'{line}\n{tree.format()}'
            recover = tree.collapse_unary().recover_unary()
            assert tree == recover, f'Unary recovery is not correct\n{tree}\n{tree.collapse_unary()}\n{recover}'
            return tree

    class Span:
        """Abstract representation of the elements of a span."""
        def __init__(self, LEFT: int, RIGHT: int, LABEL: str):
            self.LEFT = LEFT 
            self.RIGHT = RIGHT 
            self.LABEL = LABEL 
            
        def __repr__(self) -> str:
            return f'Span(LEFT={self.LEFT}, RIGHT={self.RIGHT}, LABEL={self.LABEL})'
        
        def __len__(self) -> int:
            return self.RIGHT - self.LEFT 
        
        def __lt__(self, other: PTB.Span) -> bool:
            if len(other) == len(self):
                return self.LEFT < other.LEFT
            else:
                return len(self) < len(other)
            
        def __eq__(self, other: PTB.Span) -> bool:
            return (self.LEFT == other.LEFT) and (self.RIGHT == other.RIGHT) and (self.LABEL == other.LABEL)
            
    def __init__(self, sens: List[PTB.Tree], name: str):
        super().__init__(sens, name)
        self._distro = None 
    
    @classmethod
    def from_file(cls, path: str, num_workers: int = 1, name: Optional[str] = None) -> PTB:
        name = name or path.split('/')[-1].split('.')[0]
        lines = [line for line in open(path, 'r').read().split('\n') if len(line) > 0]
        trees = parallel(cls.Tree.from_ptb, lines, num_workers=num_workers, name=name)
        return cls(trees, name) 
    
    @property
    def CONSTITUENT(self) -> str:
        values, freqs = np.unique(flatten(tree.constituents for tree in self), return_counts=True)
        return values[np.argmax(freqs)]
    

    
