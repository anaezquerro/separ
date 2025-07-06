
import torch 

from separ.data.tkz.input import InputTokenizer
from separ.utils import invert_dict, Config

    
class TargetTokenizer(InputTokenizer):
    
    def __init__(
        self,
        name: str,
        field: str | None = None,
        vocab_size: int | None = None,
        pad_token: str = InputTokenizer.PAD_TOKEN,
        unk_token: str = InputTokenizer.UNK_TOKEN,
        lower: bool = False
    ):
        """
        Instantiate a tokenizer for target data. Target tokenizers only have a padding token.
        
        Args:
            name (str): Name of the tokenizer.
            field (str): Field associated to load data.
            vocab_size (int | None): Maximum number of tokens.
            pad_token (str): Padding token.
            lower (bool): Whether to store only lowercase tokens.
        """
        self.name = name 
        self.field = field or name
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token 
        self.lower = lower
        self.bos, self.eos = False, False
        
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        self.counter = dict() # store the frequency of terms
        self.inv_vocab = invert_dict(self.vocab)

    def __repr__(self) -> str:
        return f'TargetTokenizer(name={self.name}, field={self.field}, n={len(self)})'
    
    @property
    def pad_index(self) -> int:
        return self.vocab[self.pad_token]
    
    @property
    def unk_index(self) -> int:
        return self.vocab[self.unk_token]
    
    @property
    def bos_index(self) -> int:
        raise NotImplementedError
    
    @property
    def eos_index(self) -> int:
        raise NotImplementedError
    
    @property
    def special_tokens(self) -> dict[str, int]:
        return {self.pad_token: self.pad_index, self.unk_token: self.unk_index}
    
    def encode(self, tokens: list[str], *_, **__) -> torch.Tensor:
        # call super class but never with BoS or EoS tokens
        return super().encode(tokens, bos=False, eos=False)
    
    @property
    def conf(self) -> Config:
        conf = super().conf 
        conf.most_frequent = self.vocab[sorted(self.vocab.keys() & self.counter.keys(), key=self.counter.get)[-1]]
        conf.weights = self.weights
        return conf 
        
    @property
    def weights(self) -> torch.Tensor:
        counts = torch.full((len(self),), sum(self.counter.values())-1)
        for token, count in self.counter.items():
            counts[self.vocab[token]] = count
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        return weights
    
    