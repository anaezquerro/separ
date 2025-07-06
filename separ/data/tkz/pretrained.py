import torch 
from transformers import AutoTokenizer 
from torch.nn.utils.rnn import pad_sequence

from separ.data.struct.dataset import Sentence 
from separ.data.tkz.input import InputTokenizer
from separ.utils.fn import flatten, max_len 
from separ.utils.config import Config 

class PretrainedTokenizer(InputTokenizer):
    def __init__(
        self, 
        pretrained: str,
        name: str, 
        field: str | None = None,
        bos: bool = False,
        eos: bool = False,
        fix_len: int = 3
    ):
        """
        HuggingFace pretrained tokenizer.
        
        Args:
            name (str): Name of the tokenizer.
            pretrained (str): HuggingFace model.
            field (str): Input field of the tokenizer.
            fix_len (str): Maximum number of indices to represent a token.
            bos (bool): Activate the BoS token in the encoding process (to deactivate, tkz.bos = False).
            eos (bool): Activate the EoS token in the encoding process (to deactivate, tkz.eos = False).
        """
        self.pretrained = pretrained 
        self.name = name 
        self.field = field or name 
        self.bos = bos 
        self.eos = eos 
        self.fix_len = fix_len
        self.tkz = AutoTokenizer.from_pretrained(pretrained)
        
        # add special tokens        
        self.pad_token = self.tkz.pad_token or InputTokenizer.PAD_TOKEN
        self.unk_token = self.tkz.unk_token or InputTokenizer.UNK_TOKEN 
        self.bos_token = self.tkz.bos_token or InputTokenizer.BOS_TOKEN
        self.eos_token = self.tkz.eos_token or InputTokenizer.EOS_TOKEN
        
        new_tokens = dict()
        for token in ['pad_token', 'unk_token', 'bos_token', 'eos_token']:
            if getattr(self.tkz, token) is None:
                # if it is not possible, add the default value of the InputTokenizer
                new_tokens[token] = getattr(InputTokenizer, token.upper())
                
        self.tkz.add_special_tokens(new_tokens)
        
    @property
    def vocab(self):
        return self.tkz.vocab
        
    def __repr__(self):
        return f'PretrainedTokenizer(name={self.name}, pretrained={self.pretrained}, field={self.field}, n={len(self)})'
    
    def __len__(self) -> int:
        return len(self.tkz)
    
    def encode(self, tokens: list[str] | Sentence, bos = False, eos = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encoding with variable indices. Note that each token might be mapped to a sequence of indices.
        
        Args:
            tokens (list[str] ~ seq_len): Sequence of tokens or sentence object.
            bos (bool): Whether to add a BoS token.
            eos (bool): Whether to add an EoS token.
            
        Returns:
            torch.Tensor ~ var_len: Indices.
            torch.Tensor: Number of indices used per token.
        """
        if isinstance(tokens, Sentence):
            tokens = getattr(tokens, self.field)
        if self.bos or bos:
            tokens = [self.bos_token] + tokens 
        if self.eos or eos:
            tokens.append(self.eos_token)
        indices = self.tkz(tokens, add_special_tokens=False, max_length=self.fix_len, truncation=True).input_ids
        lens = torch.tensor(list(map(len, indices)))
        return torch.tensor(flatten(indices)), lens
    
    def batch_encode(
        self, 
        batch: list[list[str]],
        bos: bool = False,
        eos: bool = False,
        mode: str = 'pad',
        pin: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Performs the encoding process of a batch of tokens.
        
        Args:
            batch (list[list[str] ~ seq_len] ~ batch_size): Batch of tokens.
            bos (bool): Whether to add the BoS index.
            eos (bool): Whether to add the EoS index.
            mode (str): Batching mode.
            
        Returns: 
            torch.Tensor ~ [batch_size, max_var_len]: Padded or concatenated batch of indices.
            list[list[int]]: Number of indices per token.
        """
        batch_indices, batch_lens = zip(*[self.encode(tokens, bos, eos) for tokens in batch])
        if mode == 'pad':
            batch_indices = pad_sequence(batch_indices, batch_first=True, padding_value=self.pad_index)
        elif mode == 'cat':
            batch_indices = torch.cat(batch_indices)
        if pin:
            batch_indices = batch_indices.pin_memory()
        return batch_indices, batch_lens
    
    def decode(self, indices: torch.Tensor) -> list[str]:
        raise NotImplementedError
    
    def train(self, tokens: list[str]):
        raise NotImplementedError
    
    @property
    def conf(self) -> Config:
        conf = super().conf 
        conf.max_len = max_len(self.pretrained, 512)
        return conf