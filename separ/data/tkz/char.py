import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from separ.utils import invert_dict, flatten
from separ.data.tkz.input import InputTokenizer
from separ.data.struct.dataset import Dataset, Sentence
    
    
class CharacterTokenizer(InputTokenizer):
    def __init__(self, *_, **__):
        super().__init__(*_, **__)
        self.lower = True 

    def __repr__(self):
        return f'CharacterTokenizer(name={self.name}, field={self.field}, n={len(self)})'
    
    def enc(self, token: str) -> torch.Tensor:
        """
        Encodes the characters of a single token 
        
        Args:
            token (str): Input token.
            
        Returns:
            torch.Tensor ~ len(token)dsf
        """
        return torch.tensor(list(map(super().enc, token)))

    def encode(
        self,
        tokens: list[str] | Sentence,
        bos: bool = False,
        eos: bool = False
    ) -> torch.Tensor:
        """
        Performs the encoding process (characters -> indices) from a sequence of tokens.
        
        Args:
            tokens (list[str] ~ seq_len): List of input tokens or sentence object.
            bos (bool): Whether to add the BoS index.
            eos (bool): Whether to add the EoS index.
            
        Returns:
            torch.Tensor ~ [seq_len, max(word_lens)]
        """
        if isinstance(tokens, Sentence):
            tokens = getattr(tokens, self.field)
        indices = list(map(self.enc, tokens))
        if self.bos or bos:
            indices = indices + [torch.tensor([self.bos_index])]
        if self.eos or eos:
            indices.append(torch.tensor([self.eos_index]))
        # now apply padding
        return pack_padded_sequence(
            pad_sequence(indices, batch_first=True, padding_value=self.pad_index),
            lengths=list(map(len, indices)), batch_first=True, enforce_sorted=False
        )
    
    def batch_encode(
        self,
        batch: list[list[str]],
        bos: bool = False,
        eos: bool = False,
        pin: bool = False
    ) -> list[torch.Tensor]:
        """
        Performs the encoding process of a batch of tokens.
        
        Args:
            batch (list[list[str] ~ seq_len] ~ batch_size): Batch of tokens.
            bos (bool): Whether to add the BoS index.
            eos (bool): Whether to add the EoS index.
        """
        batch_indices = [self.encode(tokens, bos, eos) for tokens in batch]
        if pin:
            batch_indices = [indices.pin_memory() for indices in batch_indices]
        return batch_indices
            
            
    def decode(self, indices: torch.Tensor) -> list[str]:
        raise NotImplementedError
    
    def batch_decode(self, indices: torch.Tensor):
        raise NotImplementedError
    
    def train(self, data: Dataset):
        """
        Train the tokenizer with new characters. Note that this is only performed once.
        
        Args:
            data (Dataset): Dataset of new tokens.
        """
        assert len(self) == len(self.special_tokens), 'The tokenizer has been already trained'
        tokens = flatten(getattr(sen, self.field) for sen in data)
        if self.lower:
            tokens = [token.lower() for token in tokens]
        for token in tokens:
            if token in self.special_tokens:
                continue 
            for char in token:
                try:
                    self.vocab[char]
                except:
                    self.vocab[char] = len(self.vocab)
        self.inv_vocab = invert_dict(self.vocab)
        