from torch import nn
from torch.nn.functional import gumbel_softmax, one_hot
import torch 


from separ.utils import Config 
from separ.modules import PretrainedEmbedding, FFN, LSTM, Embedding

class GATModel(nn.Module):
    """
    Generative Adversarial Tagging model, inspired in [Tong et al., (2024)](https://aclanthology.org/2024.emnlp-main.593/).
    """
    
    def __init__(
        self, 
        gen_conf: Config, 
        dis_conf: Config,
        *target_confs: Config,
    ):
        super().__init__()
        self.confs = [gen_conf, dis_conf, *target_confs]
        self.gen_conf = gen_conf
        self.dis_conf = dis_conf
        self.weight = 1
        
        self.G_encoder = PretrainedEmbedding(**gen_conf)
        for conf in target_confs:
            self.__setattr__(f'G_{conf.name.lower()}', FFN(self.G_encoder.embed_size, conf.vocab_size))
        self.__setattr__(f'D_word_embed', Embedding(dis_conf.vocab_size, dis_conf.hidden_size, dis_conf.pad_index))
        for conf in target_confs:
            self.__setattr__(f'D_{conf.name.lower()}', LSTM(
                dis_conf.hidden_size + conf.vocab_size, dis_conf.hidden_size, num_layers=2, bidirectional=True, output_size=2
            ))
        self.target_confs = target_confs
        self.criteria = nn.CrossEntropyLoss()
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n' + \
            f'\tGen(pretrained={self.gen_conf.pretrained}, finetune={self.gen_conf.finetune}, embed_size={self.G_encoder.embed_size})\n' +\
            f'\tDis(embed_size={self.dis_conf.hidden_size})\n' +\
            '\n'.join(f'\tD_{conf.name.lower()}(vocab_size={conf.vocab_size})' for conf in self.target_confs) + '\n)'
            
    def forward(
        self,
        words: torch.Tensor,
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        Args:
            words (torch.Tensor ~ [batch_size, max(seq_lens), fix_len]): Input words. 
            targets (list[torch.Tensor ~ sum(seq_lens)] ~ n_targets): Flattened batch of targets. 
            mask (torch.Tensor ~ [batch_size, max(seq_lens)]): Padding mask. 

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generator and discriminator losses.
            list[torch.Tensor]: Predicted scores.
        """
        G_embed = self.G_encoder(words)
        D_embed = self.D_word_embed(words)
        return self.decode(G_embed, D_embed, masks, targets)
        
    def decode(
        self, 
        G_embed: torch.Tensor, 
        D_embed: torch.Tensor,
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        g_loss, d_loss, preds = 0, 0 , []
        for conf, target, mask in zip(self.target_confs, targets, masks):
            d_proj = self.__getattr__(f'D_{conf.name.lower()}')
            
            logit = one_hot(target, num_classes=conf.vocab_size)*1.0
            logit += gumbel_softmax(logit, tau=1, dim=-1) # [batch_size, seq_len, n_tags]
            logit /= 2
            
            target_embed = d_proj(torch.cat([D_embed, logit], -1))[mask]
            target = target[mask]
            ones = torch.ones_like(target, dtype=torch.long)
            d_loss += self.criteria(target_embed, ones)

            G_pred = self.__getattr__(f'G_{conf.name.lower()}')(G_embed)
            preds.append(G_pred[mask])
            pred_embed = d_proj(torch.cat([D_embed, G_pred], -1))[mask]
            d_loss += self.criteria(pred_embed, (G_pred[mask].argmax(-1) == target).long())
            g_loss += self.criteria(pred_embed, ones) + self.criteria(G_pred[mask], target)
        return [g_loss, d_loss], preds
        
    def predict(self, words: torch.Tensor, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        embed = self.G_encoder(words)
        scores = [self.__getattr__(f'G_{conf.name.lower()}')(embed[mask]) for conf, mask in zip(self.target_confs, masks)]
        return self.postprocess(scores)
            
    def postprocess(self, scores: list[torch.Tensor]) -> list[torch.Tensor]:
        preds = []
        for score, conf in zip(scores, self.target_confs):
            score = torch.nan_to_num(score, 0)
            score[:, conf.special_indices] = score.min()-1
            preds.append(score.argmax(-1))
        return preds
    