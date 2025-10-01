from __future__ import annotations
from argparse import ArgumentParser
import torch, logging, time, os 

from separ.models.tag.tagger import Tagger
from separ.models.tag.dit.model import DiTModel
from separ.data import Sentence, Dataset, PretrainedTokenizer, TargetTokenizer, InputTokenizer
from separ.utils import get_mask, ControlMetric, TaggingMetric, acc, Config, is_main, logger, bar, to

class DiTagger(Tagger):
    NAME = 'ditag'
    MODEL = DiTModel
    
    def __init__(
        self,
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        device: int
    ):
        super().__init__(input_tkzs, target_tkzs, model_confs, device)
    
    def collate(self, sens: list[Sentence]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[Sentence]]:
        inputs = [tkz.batch_encode(sens, pin=False) for tkz in self.input_tkzs]
        targets = [tkz.batch_encode(sens, pin=False) for tkz in self.target_tkzs]
        mask = get_mask(list(map(len, sens)))
        return inputs, [mask.clone() for _ in targets], targets, sens 
    
    def train_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        loss = self.model.loss(*self.model(*inputs, masks, targets))
        return loss, ControlMetric(loss=loss.detach())
    
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        sens: list[Sentence]
    ) -> list[Sentence]:
        preds = self.model.predict(*inputs, masks)
        return map(self._pred, sens, *[pred.split(mask.sum(-1).tolist()) for pred, mask in zip(preds, masks)])

    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        _: list[Sentence]
    ) -> tuple[ControlMetric, TaggingMetric, list[Sentence]]:
        loss = self.model.loss(*self.model(*inputs, masks, targets))
        preds = self.model.predict(*inputs, masks)
        targets = [target[mask] for target, mask in zip(targets, masks)]
        return ControlMetric(loss=loss.detach(), \
            **dict(zip(self.TARGET_FIELDS, map(acc, preds, targets)))
        ), self.METRIC(preds, targets), []
        
    @torch.no_grad()
    def prog_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        _
    ) -> list[TaggingMetric]:
        preds = list(zip(*self.model.progression(*inputs, masks)))
        targets = [target[mask] for target, mask in zip(targets, masks)]
        metrics = [TaggingMetric(self.target_tkzs)(pred, targets) for pred in preds]
        return metrics 
    
    def progression(
        self, 
        data: str | Dataset,
        output_folder: str | None = None,
        batch_size: int = 100,
        log: logging.Logger | None = None,
        **_
    ) -> list[TaggingMetric]:
        self.barrier()
        if is_main() and output_folder:
            os.makedirs(output_folder, exist_ok=True)
        if log is None and is_main():
            log = logger('prog', path=f'{output_folder}/prog.log' if output_folder else None, level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Evaluation from {data}')
        data = self.transform_data(data)
        self.model.eval()
        start = time.time()
        loader, sampler = self.loader(data, batch_size, shuffle=True)
        metrics = [TaggingMetric(self.target_tkzs) for _ in range(self.T)]
        with bar(desc=f'eval', total=sampler.num_sens, leave=False, disable=not is_main()) as pbar:
            for *batch, sens in loader:
                _metrics = self.prog_step(*to(self.device, *batch), sens)
                for t, _metric in enumerate(_metrics):
                    metrics[t] += _metric 
                pbar.update(len(sens))
                self.barrier()
        elapsed = time.time()-start
        for metric in metrics:
            metric.to(self.device).sync()
        if log:
            log.info(f'{data.name}: {metric} {data.n_tokens/elapsed:.2f} tokens/s {len(data)/elapsed:.2f} sens/s]')
            if output_folder and is_main():
                for t, metric in zip(range(self.T, 0, -1), metrics):
                    metric.save(f'{output_folder}/{data.name}-{t}.mt')
        return metrics
        
    @classmethod
    def build(
        cls,
        data: str | Dataset,
        fields: list[str],
        enc_conf: Config,
        word_conf: Config,
        dec_conf: Config,
        device: int,
        **_
    ) -> DiTagger:
        if isinstance(data, str):
            data = cls.load_data(data)
            
        word_tkz = PretrainedTokenizer(word_conf.pretrained, 'WORD', 'FORM')
        word_conf |= word_tkz.conf
        
        target_tkzs, target_confs = [], []
        for i, field in enumerate(fields):
            tkz = TargetTokenizer(f'TAG{i+1}' if len(fields) > 1 else 'TAG', field)
            tkz.train(data)
            target_tkzs.append(tkz)
            target_confs.append(tkz.conf)
            
        return cls([word_tkz], target_tkzs, [enc_conf, word_conf, dec_conf, *target_confs], device)