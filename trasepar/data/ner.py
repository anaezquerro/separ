from __future__ import annotations

from trasepar.structs.dataset import AbstractDataset
from trasepar.utils import parallel 
from typing import List, Optional, Tuple 
import numpy as np
from datasets import load_dataset

class NER(AbstractDataset):
    SEP = '\n\n'
    EXTENSION = 'ner'
    STORED = ['jnlpba', 'drugs', 'bionlp']
    
    class Sentence:
        FIELDS = ['FORM', 'NER']
        
        def __init__(
            self, 
            FORM: List[str], 
            NER: List[str],
            ID: Optional[int] = None
        ):
            self.FORM = list(FORM)
            self.NER = [str(ner) for ner in NER]
            self.ID = ID 
            self._transformed = False
            
        def __repr__(self) -> str:
            return 'Sentence(\n\t'  + ' '.join(self.FORM) + '\n\t' + ' '.join(self.NER) + '\n)'
        
        def __len__(self) -> int:
            return len(self.FORM)
        
        def __lt__(self, other: NER.Sentence) -> bool:
            if isinstance(other, NER.Sentence):
                return self.ID < other.ID 
            else:
                raise NotImplementedError

        def __eq__(self, other: NER.Sentence):
            return all(f1 == f2 for f1, f2 in zip(self.FORM, other.FORM)) and \
                all(t1 == t2 for t1, t2 in zip(self.NER, other.NER))
                
        def format(self) -> str:
            return f'\n'.join(f'{form}\t{ner}' for form, ner in zip(self.FORM, self.NER))
        
        def copy(self) -> NER.Sentence:
            return self.__class__(self.FORM.copy(), self.NER.copy(), self.ID)
        
        @classmethod
        def from_simple(self, raw) -> NER.Sentence:
            return NER.Sentence(raw['tokens'], raw['ner_tags'])
        
        @classmethod
        def from_bio(cls, raw) -> NER.Sentence:
            tokens = raw['text'].strip().split()
            lens = np.array(list(map(len, tokens)))
            offsets = np.concatenate([np.array([0]), np.cumsum(lens+1), np.array([len(raw['text'].strip())])])
            tags = np.array(['<none>' for _ in tokens])
            for tag in raw['text_bound_annotations']:
                start = 0
                while tag['offsets'][0][0] not in range(offsets[start], offsets[start+1]):
                    start += 1
                n = len(tag['text'][0].split())
                tags[start:(start+n)] = tag['type']
            return NER.Sentence(tokens, tags)
        
        @classmethod
        def from_raw(self, raw: str) -> NER.Sentece:
            lines = raw.strip().split('\n')
            tokens, tags = zip(*[line.split('\t') for line in lines])
            return NER.Sentence(tokens, tags)
        
    @classmethod 
    def from_file(cls, path: str, num_workers: int = 1) -> NER:
        blocks = open(path, 'r').read().split('\n\n')
        sens = parallel(NER.Sentence.from_raw, blocks, num_workers=num_workers, name=path.split('/')[-1])
        return cls(sens, path)
            
    @classmethod
    def from_repo(cls, name: str) -> Tuple[NER, NER, NER]:
        assert name in cls.STORED, f'Dataset is not valid: {name}'
        if name == 'jnlpba':
            data = load_dataset('jnlpba', trust_remote_code=True)
            train = NER(list(map(NER.Sentence.from_simple, data['train'])), path=None)
            test = NER(list(map(NER.Sentence.from_simple, data['validation'])), path=f'treebanks/{name}/test.ner')
            train, dev = train.split(p=0.1)
            train.path = f'treebanks/{name}/train.ner'
            dev.path = f'treebanks/{name}/dev.ner'
        elif name == 'drugs':
            data = load_dataset("pnr-svc/Drugs-NER-Data")
            train = NER(list(map(NER.Sentence.from_simple, data['train'])), path=f'treebanks/{name}/train.ner')
            dev = NER(list(map(NER.Sentence.from_simple, data['validation'])), path=f'treebanks/{name}/dev.ner')
            test = NER(list(map(NER.Sentence.from_simple, data['test'])), path=f'treebanks/{name}/test.ner')
        elif name == 'bionlp':
            data = load_dataset('bigbio/bionlp_st_2013_cg', trust_remote_code=True)
            train = NER(list(map(NER.Sentence.from_bio, data['train'])), path=f'treebanks/{name}/train.ner')
            dev = NER(list(map(NER.Sentence.from_bio, data['validation'])), path=f'treebanks/{name}/dev.ner')
            test = NER(list(map(NER.Sentence.from_bio, data['test'])), path=f'treebanks/{name}/test.ner')
        return train, dev, test
            