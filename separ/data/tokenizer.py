from typing import Optional, List, Union, Tuple, Dict 
import torch, pickle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from transformers import AutoTokenizer

from separ.utils import onehot, flatten, pad2D, Config, max_len 


class AbstractTokenizer:
    """General implementation of an abstract simple tokenizer."""
    
    EXTENSION: str = None
    TRAINABLE: bool = False
    PARAMS: List[str] = None 
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'

    def __init__(
        self, 
        name: str, 
        field: Optional[str] = None, 
        pad_token: Optional[str] = '<pad>',
        unk_token: Optional[str] = '<unk>',
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        max_words: Optional[int] = None,
        lower: bool = False,
        packed: bool = False
    ):
        self.name = name
        self.field = field if field is not None else name
        self.pad_token = pad_token 
        self.unk_token = unk_token 
        self.bos_token = bos_token 
        self.eos_token = eos_token 
        self.max_words = max_words
        self.lower = lower 
        self.packed = packed 
        
        self.special_tokens: List[str] = [token for token in (pad_token, unk_token, bos_token, eos_token) if token is not None]
        self.counter: Dict[str, int] = dict()
        self.reset()
        
    def __len__(self) -> int:
        return len(self.vocab)
    
    def reset(self):
        """Reset vocabulary maintaining special tokens."""
        self.vocab: Dict[str, int] = {token: i for i, token in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {index: token for token, index in self.vocab.items()}
        
    def preprocess(self, token: str) -> str:
        return token.lower() if self.lower else token 
    
    def add(self, token: str) -> int:
        """Add a new token to the vocabulary. If it already exists, do nothing."""
        token = self.preprocess(token)
        try:
            return self.vocab[token]
        except KeyError:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[self.vocab[token]] = token 
            return self.vocab[token]
        
    def count(self, token: str):
        """Increase the count of a new token."""
        token = self.preprocess(token)
        try:
            self.counter[token] += 1 
        except KeyError:
            self.counter[token] = 1 
            
    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        """Get index or token for a given input."""
        if isinstance(item, str):
            try:
                return self.vocab[item]
            except KeyError:
                return self.unk_index
        else:
            return self.inv_vocab[item]
        
    def train(self, *tokens):
        """Train tokenizer from a given set of input tokens. The training process associates each 
        token to a new index. If the TRAINABLE flag is activated, it also updates the counter 
        dictionary and maintain the most occurrent tokens.
        
        Args:
            tokens (List[str]): Token dataset.
        """
        for token in tokens:
            self.count(token)
        if self.TRAINABLE and self.max_words is not None:
            tokens = sorted(self.counter.keys(), key=self.counter.get, reverse=True)[:self.max_words]
        if self.TRAINABLE:
            for token in tokens:
                self.add(token)
            
    def remove_special(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Skip special tokens from an input batch. 

        Args:
            batch (torch.Tensor ~ [batch_size, max(seq_len)]): Input batch.

        Returns: 
            Tuple[
                torch.Tensor ~ [batch_size * seq_len]: Filtered batch.
                torch.Tensor ~ [batch_size]: Number of non-special indices per sentence.
            ]
        """
        mask = torch.ones_like(batch, dtype=torch.bool)
        for index in self.special_indices:
            mask &= (batch != index)
        return batch[mask], mask.sum(-1)
            
    def encode(self, tokens: List[str]) -> torch.Tensor:
        """Sentence-level encoding.

        Args:
            tokens (List[str] ~ [seq_len]): Input tokens (assumed to be a sentence).

        Raises:
            NotImplementedError: Meant to be defined in subclasses.

        Returns:
            torch.Tensor ~ [seq_len]: Encoded indices.
        """
        raise NotImplementedError
            
    def batch_encode(self, batch: List[List[str]]) -> Union[torch.Tensor, PackedSequence]:
        """Batch-level encoding.

        Args:
            batch (List[List[str]] ~ [batch_size, seq_len]): List of input tokens (assumed to be a batch).

        Returns:
            Union[torch.Tensor, PackedSequence] ~ [batch_size, max(seq_len)]: Encoded batch.
        """
        inputs = [self.encode(tokens) for tokens in batch]
        return self.batch(inputs)
    
    def batch(self, inputs: List[torch.Tensor]) -> Union[torch.Tensor, PackedSequence]:
        """Batchification form a list of tensor inputs.

        Args:
            inputs (List[torch.Tensor] ~ [batch_size]): List of tensor inputs.

        Returns:
            Union[torch.Tensor, PackedSequence] ~ [batch_size, max(seq_len)]: Padded batch.
        """
        batch = self.pad(inputs)
        if self.packed:
            batch = self.pack(batch)
        return batch
        
    
    def decode(self, inputs: torch.Tensor, remove_special: bool = False) -> List[str]:
        """Sentence-level decoding.

        Args:
            inputs (torch.Tensor ~ [seq_len]): Input indices (assumed to be a sentence).
            remove_special (bool, optional): Whether to remove special tokens. Defaults to False.

        Raises:
            NotImplementedError: Meant to be defined in subclasses.

        Returns:
            List[str] ~ [seq_len]: Decoded tokens.
        """
        raise NotImplementedError
    
    def batch_decode(self, inputs: torch.Tensor, remove_special: bool = True) -> List[List[str]]:
        """Batch-level decoding.

        Args:
            inputs (torch.Tensor ~ [batch_size, max(seq_len)]): Input batch.
            remove_special (bool, optional): Whether to remove special tokens. Defaults to True.

        Returns:
            List[List[str]] ~ [batch_size, seq_len]: Decoded batch.
        """
        if remove_special:
            inputs, lens = self.remove_special(inputs)
            inputs = inputs.split(lens.tolist())
        else:
            inputs = inputs.unbind(0)
        return list(map(self.decode, inputs))
    
    def pad(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(tensors, True, self.pad_index)
    
    def pack(self, inputs: torch.Tensor) -> PackedSequence:
        lens = (inputs != self.pad_index).sum(-1)
        return pack_padded_sequence(inputs, lengths=lens, batch_first=True, enforce_sorted=False)
    
    def enc(self, token: str) -> int:
        try:
            return self.vocab[self.preprocess(token)]
        except KeyError:
            return self.unk_index
        
    def dec(self, index: int) -> str:
        return self.inv_vocab[index]
        
    @property
    def pad_index(self) -> int:
        return self.vocab[self.pad_token]
    
    @property
    def unk_index(self) -> int:
        return self.vocab[self.unk_token]
    
    @property
    def bos_index(self) -> int:
        return self.vocab[self.bos_token]
    
    @property
    def eos_index(self) -> int:
        return self.vocab[self.eos_token]
    
    @property 
    def most_frequent(self) -> int:
        return self.vocab[sorted(self.counter.keys(), key=self.counter.get)[-1]]
    
    @property 
    def special_indices(self) -> List[int]:
        indices = []
        if self.pad_token is not None:
            indices.append(self.pad_index)
        if self.unk_token is not None:
            indices.append(self.unk_index)
        if self.bos_token is not None:
            indices.append(self.bos_index)
        if self.eos_token is not None:
            indices.append(self.eos_index)
        return indices
    
    @property
    def conf(self) -> Config:
        return Config(vocab_size=len(self), 
                      pad_index=self.pad_index if self.pad_token is not None else None, 
                      unk_index=self.unk_index if self.unk_token is not None else None,
                      special_indices=self.special_indices,
                      name=self.name.lower()
                      )
        
    def save(self, path: str):
        if not path.endswith(self.EXTENSION):
            path += f'.{self.EXTENSION}'
        objects = {param: getattr(self, param) for param in self.PARAMS}
        with open(path, 'wb') as writer:
            pickle.dump(objects, writer)
    
    @classmethod
    def load(cls, path): 
        with open(path, 'rb') as reader:
            params = pickle.load(reader)
        counter = params.pop('counter') if 'counter' in params.keys() else None 
        vocab = params.pop('vocab') if 'vocab' in params.keys() else None
        if 'name' not in params.keys():
            params['name'] = path.split('/')[-1].split('.')[0]
        tkz = cls(**params)
        if counter is not None:
            tkz.counter = counter
        if vocab is not None:
            tkz.vocab = vocab 
            tkz.inv_vocab = {idx: token for token, idx in vocab.items()}
        return tkz 
    

class Tokenizer(AbstractTokenizer):
    EXTENSION: str = 'tkz'
    TRAINABLE: bool = True 
    PARAMS: List[str] = ['name', 'field', 'pad_token', 'unk_token', 'bos_token', 'eos_token', 'lower', 'max_words', 'packed', 'counter', 'vocab']

    def encode(self, tokens: List[str]) -> torch.Tensor:
        indices = []
        if self.bos_token:
            indices.append(self.bos_index)
        for token in tokens:
            try:
                indices.append(self.vocab[self.preprocess(token)])
            except KeyError:
                indices.append(self.unk_index)
        if self.eos_token:
            indices.append(self.eos_index)
        return torch.tensor(indices)
    
    def decode(self, indices: torch.Tensor, remove_special: bool = False) -> List[str]:
        if remove_special:
            indices, _ = self.remove_special(indices)
        return list(map(self.inv_vocab.get, indices.tolist()))
    
    @property
    def conf(self) -> Config:
        conf = super().conf 
        conf.most_frequent = self.most_frequent
        return conf 
    


class PretrainedTokenizer(AbstractTokenizer):
    EXTENSION = 'tkz-pret'
    TRAINABLE = False 
    PARAMS = ['name', 'field', 'pretrained', 'lower', 'bos', 'eos', 'fix_len', 'counter']
    
    def __init__(
        self, 
        name: str,
        field: str, 
        pretrained: str, 
        lower: bool = False, 
        bos: bool = False, 
        eos: bool = False,
        fix_len: int = 3,
    ):
        self.name = name
        self.field = field 
        self.pretrained = pretrained
        self.lower = lower 
        self.bos = bos 
        self.eos = eos 
        self.fix_len = fix_len 
        self.tkz = AutoTokenizer.from_pretrained(pretrained, padding_side='right')
        self.counter = dict()
        self.max_len = max_len(pretrained)
        self.pad_token = self.tkz.pad_token or self.tkz.eos_token 
        self.unk_token = self.tkz.unk_token 
        self.bos_token = self.tkz.bos_token or self.tkz.cls_token 
        self.eos_token = self.tkz.eos_token or self.tkz.sep_token 
        self.packed = False
        
        self.args = dict(padding='max_length', add_special_tokens=False, max_length=fix_len, return_tensors='pt', truncation=True)
        
        if self.pad_token in [self.unk_token, self.bos_token, self.eos_token]:
            self.tkz.add_special_tokens({'pad_token': '<pad>'})
            self.pad_token = '<pad>'
        
        assert self.pad_token not in [self.unk_token, self.bos_token, self.eos_token], \
            f'pad_token ({self.pad_token}) must be different than other special tokens'
        
    def encode(self, tokens: List[str], flattened: bool = False) -> torch.Tensor:
        if self.bos:
            tokens = [self.bos_token] + tokens 
        if self.eos:
            tokens.append(self.eos_token)
        inputs = self.tkz(tokens, **self.args).input_ids
        if flattened:
            inputs = inputs[inputs != self.pad_index]
        return inputs 
    
    def decode(self, encoded: torch.Tensor, remove_special: bool = True) -> str:
        return self.tkz.decode(encoded, skip_special_tokens=remove_special)
    
    def batch_decode(self, encoded: torch.Tensor, remove_special: bool = True) -> List[str]:
        return self.tkz.batch_decode(encoded, skip_special_tokens=remove_special)
    
    @property
    def pad_index(self) -> int:
        return self.tkz.convert_tokens_to_ids(self.pad_token)
    
    @property
    def unk_index(self) -> int:
        return self.tkz.convert_tokens_to_ids(self.unk_token)
    
    @property
    def bos_index(self) -> int:
        return self.tkz.convert_tokens_to_ids(self.bos_token)
    
    @property
    def eos_index(self) -> int:
        return self.tkz.convert_tokens_to_ids(self.eos_token)
    
    def __len__(self) -> int:
        return len(self.tkz)
    
    @property 
    def conf(self) -> Config:
        conf = super().conf 
        conf.max_len = self.max_len
        return conf 
    
    @property 
    def most_frequent(self) -> int:
        return torch.randint(len(self), size=(1,)).item()
        

        
    
class OneHotTokenizer(Tokenizer):
    """Implements the one-hot encoding for input tokens."""
    
    EXTENSION = 'tkz-one'
    TRAINABLE = True
    PARAMS = ['name', 'field', 'unk_token', 'counter', 'vocab']
    
    def __init__(self, name: str, field: Optional[str] = None, unk_token: str = '<unk>'):
        super().__init__(name, field, pad_token=None, unk_token=unk_token, bos_token=None, 
                         eos_token=None, max_words=None, lower=False, packed=False)
    
    def onehot(self, tokens: List[str]) -> torch.Tensor:
        indices = []
        for token in tokens:
            try:
                indices.append(self.vocab[self.preprocess(token)])
            except KeyError:
                indices.append(self.unk_index)
        return onehot(indices, len(self))
    
    def encode(self, tokens: List[List[str]]) -> torch.Tensor:
        """Performs one-hot encoding for each element of the token list.

        Args:
            tokens (List[List[str]]): Nested list of tokens. Each item is a subset of tokens that 
            will be represented with a one-hot vector.

        Returns:
            torch.Tensor: Sequence of one-hot vectors.
        """
        encoded = torch.zeros(len(tokens), len(self), dtype=torch.bool)
        for i, token in enumerate(tokens):
            indices = list(map(self.enc, token))
            encoded[i, indices] = True 
        return encoded 
    
    def batch(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return pad2D(inputs)
    
    def __decode(self, encoded: torch.Tensor, remove_special: bool = True) -> List[str]:
        assert len(encoded.shape) == 1, 'The one-hot encoding tensor must be a vector'
        if remove_special:
            encoded[self.special_indices] = False
        tokens = [self.inv_vocab[index] for index in encoded.nonzero().flatten().tolist()]
        return tokens 
    
    def decode(self, encoded: torch.Tensor, remove_special: bool = False) -> List[List[str]]:
        """Decodes a sequence of one-hot encoded vectors.

        Args:
            encoded (torch.Tensor): ``[seq_len, vocab_size]``.
            remove_special (Optional[str], optional): Whether to replace special tokens. Defaults to None.

        Returns:
            List[List[str]]: Decoded tokens.
        """
        assert encoded.shape[-1] == len(self), f'Encoded input must have dimension vocab_size={len(self)}'
        return [self.__decode(x, remove_special) for x in encoded.unbind(0)]
    
    def batch_decode(self, batch: List[torch.Tensor], remove_special: bool = True) -> List[List[List[str]]]:
        return [self.decode(encoded, remove_special) for encoded in batch]
    
            
        
class CharacterTokenizer(AbstractTokenizer):
    EXTENSION = 'char-tkz'
    TRAINABLE = True
    PARAMS = ['name', 'field', 'pad_token', 'unk_token', 'bos_token', 'eos_token', 'lower', 'counter', 'vocab']
    
    def __init__(
        self, 
        name: str, 
        field: Optional[str] = None, 
        pad_token: Optional[str] = '<pad>',
        unk_token: Optional[str] = '<unk>',
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        lower: bool = False
    ):
        super().__init__(name, field, pad_token=pad_token, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, max_words=None, lower=lower, packed=True)
    
    def train(self, *tokens):
        super().train(*flatten(list(token) for token in tokens))
        
    def _encode(self, token: str) -> torch.Tensor:
        token = self.preprocess(token)
        indices = []
        for char in token:
            try:
                indices.append(self.vocab[char])
            except KeyError:
                indices.append(self.unk_index)
        return torch.tensor(indices)

    def encode(self, tokens: List[str]) -> PackedSequence:
        """Sentence-level encoding. 

        Args:
            tokens (List[str]): Input tokens (assumed to be a sentence).

        Returns:
            PackedSequence ~ [seq_len, max(token_len)]: Character indices in a padded tensor.
        """
        indices = list(map(self._encode, tokens))
        if self.bos_token:
            indices = [torch.tensor([self.bos_index])] + indices 
        if self.eos_token:
            indices += [torch.tensor([self.eos_index])]
        return self.pack(self.pad(indices))
    
    def batch(self, inputs: List[PackedSequence]) -> List[PackedSequence]:
        return inputs
    
    