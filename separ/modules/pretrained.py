from torch import nn 
from transformers import AutoModel
import torch 
from torch.nn.utils.rnn import pad_sequence

from separ.utils.fn import max_len

def agg(embed: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Aggregates embeddings positions in a sequence matrix.

    Args:
        embed (torch.Tensor ~ [m_seq_len, hidden_size]): Multi-token embeddings matrix.
        lens (torch.Tensor ~ seq_len): Number of embeddings that are aggregated at each position.
            Note that in_seq_len >= seq_len and sum(lens) = in_seq_len.

    Returns:
        torch.Tensor ~ [seq_len, hidden_size]: Aggregated sequence embedding matrix.
    """
    end = lens.cumsum(0).tolist()
    start = [0] + end[:-1]
    return torch.stack([embed[s:e].mean(0) for s, e in zip(start, end)])

class PretrainedEmbedding(nn.Module):
    
    def __init__(
        self, 
        pretrained: str, 
        pad_index: int, 
        vocab_size: int,
        finetune: bool,
        max_len: int,
        **_,
    ):
        """Pretrained embedding from a HuggingFace model.
        
        Args:
            pretrained (str): Pretrained weights (from https://huggingface.co/models).
            pad_index (int): Padding index.
            vocab_size (int): New vocabulary size (resize if needed).
            finetune (bool): Finetune model.
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.pretrained = pretrained
        self.finetune = finetune
        self.embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.embed.resize_token_embeddings(vocab_size)
        self.embed_size = self.embed.config.hidden_size
        self.pad_index = pad_index
        self.max_len = max_len
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pretrained={self.pretrained}, embed_size={self.embed_size}, finetune={self.finetune}, max_len={self.max_len})'

    def forward(self, x: tuple[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """Returns contextualized embeddings.

        Args:
            words (torch.Tensor ~ [batch_size, max(m_seq_lens)]): Padded multi-token inputs.
            lens (list[torch.Tensor ~ seq_len] ~ batch_size): List of lengths for each sentence.

        Returns:
            torch.Tensor ~ [batch_size, max(seq_lens)]: Embedding matrix.
        """
        
        words, lens = x
        mask = (words != self.pad_index).to(torch.int32)
        if words.shape[-1] >= self.max_len:
            embed = torch.cat([
                self.embed(words[:, i:(i+self.max_len)], attention_mask=mask[:, i:(i+self.max_len)]).last_hidden_state
                for i in range(0, words.shape[1], self.max_len)
            ], 1)
        else:
            embed = self.embed(words, attention_mask=mask).last_hidden_state
        embed = list(map(agg, embed.unbind(0), lens))
        embed = pad_sequence(embed, batch_first=True)
        return embed
    


