from torch import nn 
from typing import Optional
import torch 

from trasepar.utils import Config 
from trasepar.modules import PretrainedEmbedding, LSTM, CharLSTM, FFN, Embedding, xLSTM

class Model(nn.Module):
    """Shared implementation of the encoder module of a neural model.""" 
    
    def __init__(
        self,
        enc_conf: Config, 
        word_conf: Config, 
        tag_conf: Optional[Config] = None,
        char_conf: Optional[Config] = None,
    ):
        super().__init__()
        self.dim = 0
        self.pos = tag_conf is not None 
        self.char = char_conf is not None

        # encoder
        if word_conf.pretrained:
            self.pretrained = True 
            self.word_embed = PretrainedEmbedding(**word_conf())
            enc_conf.dropout = enc_conf.dropout or 0.0
            self.encoder = FFN(self.word_embed.embed_size, enc_conf.hidden_size, dropout=enc_conf.dropout) if enc_conf.hidden_size else nn.Identity()
            self.hidden_size = enc_conf.hidden_size or self.word_embed.embed_size
        else:
            self.pretrained = False
            self.word_embed = Embedding(word_conf.vocab_size, word_conf.embed_size, word_conf.pad_index)
            self.dim = word_conf.embed_size
                    
            if tag_conf is not None:
                self.tag_embed = Embedding(tag_conf.vocab_size, tag_conf.embed_size, tag_conf.pad_index)
                self.dim += tag_conf.embed_size
            if char_conf is not None:
                self.char_embed = Embedding(char_conf.vocab_size, char_conf.embed_size, char_conf.pad_index)
                self.char_lstm = CharLSTM(char_conf.embed_size)
                self.dim += char_conf.embed_size
            
            self.hidden_size = enc_conf.hidden_size or self.dim
            self.encoder = (xLSTM if enc_conf.xlstm else LSTM)(self.dim, **enc_conf()) 
                
        self.enc_conf, self.word_conf, self.tag_conf, self.char_conf = enc_conf, word_conf, tag_conf, char_conf
            
            
    def encode(self, words: torch.Tensor, *args) -> torch.Tensor:
        """Encode batched input.

        Args:
            words (torch.Tensor): Word inputs.
                [batch_size, max(seq_len)] if not pretrained.
                [batch_size, max(seq_len), fix_len] if pretrained.
            tags (torch.Tensor ~ [batch_size, max(seq_len)]): PoS tag inputs.
            chars (List[PackedSequence ~ [seq_len, max(token_len)]] ~ [batch_size]): Character inputs.

        Returns:
            torch.Tensor ~ [batch_size, max(seq_len)]: Contextualized batch.
        """
        args = list(args)
        word_embed = self.word_embed(words)
        feats = []
        if self.pos:
            feats.append(self.tag_embed(args.pop(0)))
        if self.char:
            chars = args.pop(0)
            feats.append(self.char_lstm([self.char_embed(char) for char in chars]))
        embed = torch.cat([word_embed] + feats, -1)
        if self.word_conf.pretrained:
            return self.encoder(embed)
        else:
            embed = self.encoder(embed)
            return embed 


        
        
        
