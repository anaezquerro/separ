
from torch import nn 
from transformers import AutoModel
import torch 
from torch.nn.utils.rnn import pad_sequence

from separ.utils import split, max_len


class PretrainedEmbedding(nn.Module):
    DEFAULT_MAX_LEN = 512
    
    def __init__(
            self, 
            pretrained: str, 
            pad_index: int, 
            vocab_size: int,
            finetune: bool, 
            defreeze_ratio: float = 0,
            **_
        ):
        super().__init__()
        self.pretrained = pretrained
        self.finetune = finetune
        self.defreeze_ratio = defreeze_ratio
        self.word_embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.word_embed.resize_token_embeddings(vocab_size)
        if defreeze_ratio > 0:
            params = list(self.word_embed.parameters())
            defreeze = int(len(params)*defreeze_ratio)
            for param in params[-defreeze:]:
                param._requires_grad = True 
        self.embed_size = self.word_embed.config.hidden_size
        self.pad_index = pad_index
        self.max_len = max_len(pretrained, self.DEFAULT_MAX_LEN)
        
    def __repr__(self) -> str:
        return f'PretrainedEmbedding(pretrained={self.pretrained}, embed_size={self.embed_size}, finetune={self.finetune}, defreeze_ratio={self.defreeze_ratio})'

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        # words ~ [batch_size, pad(seq_len), fix_len]
        # mask ~ [batch_size, pad(seq_len)]
        mask = (words != self.pad_index).sum(-1) > 0
        lens = mask.sum(-1).tolist()

        # flat ~ [batch_size, pad(seq_len)]
        fmask = words != self.pad_index
        flat = pad_sequence(words[fmask].split(fmask.sum((-2, -1)).tolist()), batch_first=True, padding_value=self.pad_index)
        if flat.shape[1] > self.max_len:
            seq_len = self.max_len//2
            x = torch.cat([
                self.word_embed(flat[:, i:(i+seq_len)], attention_mask=(flat[:, i:(i+seq_len)] != self.pad_index).to(torch.int32)).last_hidden_state
                for i in range(0, flat.shape[1], seq_len)
            ], 1)
        else:
            x = self.word_embed(flat, attention_mask=(flat != self.pad_index).to(torch.int32)).last_hidden_state

        word_lens = fmask.sum(-1).flatten()
        word_lens = word_lens[word_lens > 0].tolist()
        x = split([torch.mean(i, dim=0) for i in x[flat != self.pad_index].split(word_lens)],
                  [l for l in lens if l > 0])
        embed = pad_sequence([torch.stack(i, dim=0) for i in x], padding_value=0, batch_first=True)

        # padding is needed
        if words.shape[1] > embed.shape[1]:
            embed = torch.concat(
                [embed, torch.zeros(embed.shape[0], words.shape[1] - embed.shape[1], embed.shape[2], device=words.device)],
                dim=1)
        if words.shape[0] > embed.shape[0]:
            embed = torch.concat(
                [embed, torch.zeros(words.shape[0] - embed.shape[0], embed.shape[1], embed.shape[2], device=words.device)],
                dim=0)
        return embed


