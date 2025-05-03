import torch, pickle 
from typing import Optional, Dict, List, Union 
from torch.nn.utils.rnn import pad_sequence

from separ.utils import invert_dict, Config, flatten 
from separ.data.struct.dataset import Sentence, Dataset 

class InputTokenizer:
    PAD_TOKEN: str = '<pad>'
    UNK_TOKEN: str = '<unk>'
    BOS_TOKEN: str = '<bos>'
    EOS_TOKEN: str = '<eos>'
    
    def __init__(
        self,
        name: str,
        field: Optional[str] = None,
        vocab_size: Optional[int] = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        bos: bool = False,
        eos: bool = False,
        lower: bool = False
    ):
        """
        Instantiate a tokenizer for input data. Input tokenizers always have a padding, unkown, BoS and EoS token.
        
        Args:
            name (str): Name of the tokenizer.
            field (str): Field associated to load data.
            vocab_size (Optional[int]): Maximum number of tokens.
            pad_token (str): Padding token.
            unk_token (str): Unkown token.
            bos_token (str): Begging-of-sentence token.
            eos_token (str): End-of-sentence token.
            bos (bool): Activate the BoS token in the encoding process (to deactivate, tkz.bos = False).
            eos (bool): Activate the EoS token in the encding process (to deactivate, tkz.eos = False).
            lower (bool): Whether to store only lowercase tokens.
        """
        self.name = name 
        self.field = field or name
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos = bos 
        self.eos = eos
        self.lower = lower
        
        self.vocab = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.counter = dict() # store the frequency of terms
        self.inv_vocab = invert_dict(self.vocab)
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __repr__(self) -> str:
        return f'InputTokenizer(name={self.name}, field={self.field}, n={len(self)})'
    
    @property
    def conf(self) -> Config:
        return Config(
            name=self.name, vocab_size=len(self), pad_index=self.pad_index, 
            special_indices=list(self.special_tokens.values())
    )
    
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
    def special_tokens(self) -> Dict[str, int]:
        return {
            self.pad_token: self.pad_index, 
            self.unk_token: self.unk_index, 
            self.bos_token: self.bos_index,
            self.eos_token: self.eos_index
        }
    
    def enc(self, token: str) -> int:
        """Encode a single token."""
        if self.lower:
            token = token.lower()
        try:
            return self.vocab[token]
        except KeyError:
            return self.unk_index 
        
    def dec(self, index: int) -> str:
        """Decode a single index."""
        return self.inv_vocab[index]

    def encode(
        self,
        tokens: Union[List[str], Sentence],
        bos: bool = False,
        eos: bool = False
    ) -> torch.Tensor:
        """
        Performs the encoding process (tokens -> indices) from a sequence of tokens.
        
        Args:
            tokens (List[str] ~ seq_len): List of input tokens or a sentence object.
            bos (bool): Whether to add the BoS index.
            eos (bool): Whether to add the EoS index.
        
        Returns:
            torch.Tensor ~ seq_len: Indices.
        """
        if isinstance(tokens, Sentence):
            tokens = getattr(tokens, self.field)
        indices = list(map(self.enc, tokens))
        if self.bos or bos:
            indices = [self.bos_index] + indices
        if self.eos or eos:
            indices.append(self.eos_index)
        return torch.tensor(indices, dtype=torch.long)
    
    def batch_encode(
        self, 
        batch: List[List[str]],
        bos: bool = False,
        eos: bool = False,
        mode: str = 'pad',
        pin: bool = False
    ) -> torch.Tensor:
        """
        Performs the encoding process of a batch of tokens.
        
        Args:
            batch (List[List[str] ~ seq_len] ~ batch_size): Batch of tokens.
            bos (bool): Whether to add the BoS index.
            eos (bool): Whether to add the EoS index.
            mode (str): Batching mode.
            
        Returns: 
            torch.Tensor ~ [batch_size, max_len]: Padded batch of indices.
        """
        batch_indices = [self.encode(tokens, bos, eos) for tokens in batch]
        if mode == 'pad':
            batch_indices = pad_sequence(batch_indices, batch_first=True, padding_value=self.pad_index)
        elif mode == 'cat':
            batch_indices = torch.cat(batch_indices)
        if pin:
            batch_indices = batch_indices.pin_memory()
        return batch_indices
    
    def decode(self, indices: torch.Tensor) -> List[str]:
        """
        Performs the decoding process (indices -> tokens).
        
        Args: 
            indices (torch.Tensor): Input indices.
        
        Returns:
            List[str]: Decoded tokens.
        """
        return list(map(self.dec, indices.detach().tolist()))
    
    def batch_decode(self, batch: torch.Tensor) -> List[List[str]]:
        """
        Performs the decoding process of a batch of indices.
        
        Args: 
            batch (torch.Tensor ~ [batch_size, max_len]): Padded batch of indices.
            
        Returns:
            List[List[str] ~ seq_len] ~ batch_size: Decoded tokens.
        """
        return list(map(self.decode, batch.unbind(0)))
    
    def train(self, data: Union[List[str], Dataset]):
        """
        Train the tokenizer with new tokens. Note that this is only performed once.
        
        Args:
            data (Dataset): Dataset of tokens.
        """
        assert len(self) == len(self.special_tokens), 'The tokenizer has been already trained'
        if isinstance(data, Dataset):
            tokens = flatten(getattr(sen, self.field) for sen in data)
        else:
            tokens = data
        if self.lower:
            tokens = [token.lower() for token in tokens]
        for token in tokens:
            if token in self.special_tokens:
                continue 
            try:
                self.counter[token] += 1 
            except:
                self.counter[token] = 1
        new_tokens = sorted(self.counter, key=self.counter.get, reverse=True)
        n_new = self.vocab_size or len(new_tokens)
        self.vocab |= dict(zip(new_tokens[:n_new], range(len(self), len(self)+n_new)))
        self.inv_vocab = invert_dict(self.vocab)
        
    def save(self, path: str):
        """Saves the tokenizer in the input path."""
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)
    