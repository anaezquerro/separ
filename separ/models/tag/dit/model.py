import numpy as np 
from torch import nn 
import torch 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from separ.modules.bidit import BiDiT
from separ.modules.scheduler import LinearScheduler, CosineScheduler
from separ.models.tag.model import TagModel
from separ.utils import Config

class DiTModel(TagModel):
    def __init__(
        self,
        enc_conf: Config,
        word_conf: Config,
        dec_conf: Config,
        *target_confs: list[Config]
    ):
        super(TagModel, self).__init__(enc_conf, word_conf, None, None)
        dec_conf.sched = (LinearScheduler(dec_conf.T) if dec_conf.sched_name == 'lin' else CosineScheduler(dec_conf.T))
        self.dec_conf = dec_conf 
        self.target_confs = target_confs
        self.num_bits = []
        for conf in target_confs:
            conf.num_bits = int(np.ceil(np.log2(conf.vocab_size))) # compute the number of bits 
            self.num_bits.append(conf.num_bits)
            self.__setattr__(conf.name.lower(), BiDiT(hidden_size=self.hidden_size, num_bits=conf.num_bits, max_len=word_conf.max_len, **dec_conf))
        
        if dec_conf.loss == 1:
            self.criteria = nn.L1Loss()
        elif dec_conf.loss == 2:
            self.criteria = nn.MSELoss()
        else:
            raise NotImplementedError('The loss type must be MSE (2) or L1 (1)')
        
    def to(self, device: torch.device):
        super().to(device)
        self.dec_conf.sched.to(device)
        return self
        
    def forward(
        self,
        words: torch.Tensor, 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass.

        Args:
            words (torch.Tensor ~ [batch_size, max(seq_lens), fix_len]): Conditional word inputs.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.
            targets (list[torch.Tensor ~ [batch_size, max(seq_lens)] ~ n_targets]): List of batched targets.

        Returns:
            preds (torch.Tensor ~ [sum(seq_lens), n_bits]): Bit predictions.
            targets (torch.Tensor ~ [sum(seq_lens), n_bits]): Bit targets.
        """
        return self.decode(self.encode(words), masks, targets)
    
    def decode(
        self,
        embed: torch.Tensor, 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoding step.
        
        Args:
            embed (torch.Tensor ~ [batch_size, max(seq_lens), hidden_size]): Contextualized word embeddings.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.
            tags (list[torch.Tensor ~ [batch_size, max(seq_lens)] ~ n_targets]): List of batched targets.

        
        Returns:
            preds (torch.Tensor ~ [sum(seq_lens), n_bits]): Bit predictions.
            targets (torch.Tensor ~ [sum(seq_lens), n_bits]): Bit targets.
        """
        preds, targets = zip(*[self.__getattr__(conf.name.lower())(embed, target, mask) for conf, target, mask in zip(self.target_confs, targets, masks)])
        return preds, targets

    def loss(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """Loss computation.

        Args:
            preds (list[torch.Tensor ~ [sum(seq_lens), n_bits]]): Bit representation of the packed predictions.
            targets (list[torch.Tensor ~ [sum(seq_lens), n_bits]]): Bit representation of the packed targets.

        Returns:
            torch.Tensor: MSE/L1 loss.
        """
        return sum(map(self.criteria, preds, targets))
    
    def predict(self, words: torch.Tensor, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Prediction step.

        Args:
            words (torch.Tensor ~ [batch_size, max(seq_lens), fix_len]): Conditional word inputs.
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask.

        Returns: 
            list[torch.Tensor ~ sum(seq_lens)] ~ n_targets.
        """
        embed = self.encode(words)
        preds = []
        for conf, mask in zip(self.target_confs, masks):
            pred = self.__getattr__(conf.name.lower()).predict(embed, mask)
            pred[pred >= conf.vocab_size] = conf.most_frequent 
            for i in conf.special_indices:
                pred[pred == i] = conf.most_frequent
            preds.append(pred)
        return preds 
    
    def progression(self, words: torch.Tensor, masks: list[torch.Tensor]):
        embed = self.encode(words)
        for conf, mask in zip(self.target_confs, masks):
            yield list(self.__getattr__(conf.name.lower()).progression(embed, mask))
            
    def plot(self) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[self.dec_conf.sched.__class__.__name__, 'tag-accuracy'])
        time = torch.arange(0, self.dec_conf.T).cuda()
        fig.add_trace(go.Scatter(x=time.cpu(), y=self.dec_conf.sched.beta(time).cpu(), showlegend=False))
        for conf in self.target_confs:
            dec = self.__getattr__(conf.name.lower())
            tags = torch.arange(0, len(conf)).reshape(1,-1).repeat(self.dec_conf.T,1).cuda()
            bits = dec.tag2bit(tags)
            e = torch.randn_like(bits)
            t =  time.reshape(-1,1,1).repeat(1,len(conf),bits.shape[-1])
            noise = dec.corrupt(bits, t, e)
            denoised = dec.bit2tag(noise)
            acc = (tags == denoised).float().mean(-1)
            fig.add_trace(go.Scatter(x=time.cpu(), y=acc.cpu(), name=conf.name), row=1, col=2)
        fig.update_layout(height=500, width=1600, template='seaborn', showlegend=False, margin=dict(t=30, l=30, r=10, b=30))
        fig.update_xaxes(title_text='timestep')
        return fig 
            
            
            