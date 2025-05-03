from __future__ import annotations
from typing import List, Union, Optional
from torch.optim import Optimizer
import torch, os, logging, time
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload, FullStateDictConfig, StateDictType, FullOptimStateDictConfig
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.optim import Optimizer
from torch import nn 

from separ.parser import Parser
from separ.modules import Embedding, PretrainedEmbedding
from separ.utils import WORLD_SIZE, Metric, flatten, filename, logger, Config, ControlMetric, to, bar
from separ.data import Dataset, InputTokenizer, TargetTokenizer

nonwrap_modules = {Embedding, PretrainedEmbedding}


def policy(
    module: nn.Module, 
    recurse: bool, 
    nonwrapped_numel: int, 
    min_num_params: int = int(1e4)
) -> bool:
    if any(isinstance(module, m) for m in nonwrap_modules):
        wrap = False
    else:
        wrap = nonwrapped_numel > min_num_params 
    return wrap

class DistributedParser(Parser):
    
    def __init__(
        self, 
        input_tkzs: List[InputTokenizer],
        target_tkzs: List[TargetTokenizer],
        model_confs: List[Config],
        device: int
    ):
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.model_confs = model_confs 
        for tkz in input_tkzs + target_tkzs:
            self.__setattr__(tkz.name, tkz)
        self.epochs = 1
        # policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={XLMRobertaLayer, GPT2Block, LlamaAttention})
        self.model = FSDP(
            self.MODEL(*model_confs).to(device),
            cpu_offload=CPUOffload(offload_params=True),
            # auto_wrap_policy=policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
        )
        self.device = device
        
    @classmethod
    def load(cls, path: str, device) -> Parser:
        # this parser does not have the model, only the state 
        parser = torch.load(path, weights_only=False, map_location='cpu')
        # build the model and load the state
        parser.device = device
        parser.model = FSDP(
            parser.MODEL(*parser.model_confs).to(device),
            cpu_offload=CPUOffload(offload_params=True),
            sharding_strategy=ShardingStrategy.FULL_SHARD
        )
        parser.load_state(parser.state)
        delattr(parser, 'state')
        return parser 
           
    def save(self, path: str):
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            state = self.model.state_dict()
        if self.device == 0:
            model = self.model 
            self.__delattr__('model')
            self.state = state 
            torch.save(self, path)
            self.__delattr__('state')
            self.model = model 
            
    def load_state(self, state: Union[str, dict]):
        dist.barrier()
        if isinstance(state, str):
            state = torch.load(state, weights_only=False).state
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT,
                                    FullStateDictConfig(rank0_only=False), 
                                    FullOptimStateDictConfig(rank0_only=False)):
            self.model.load_state_dict(state)
    
    def loader(self, data: Dataset, batch_size: int, shuffle: bool):
        return data.loader(batch_size=batch_size, shuffle=shuffle, collate=self.collate, device=self.device)
        
    def epoch(
        self, 
        epoch: int,
        optimizer: Optimizer,
        loader: DataLoader,
        sampler: Sampler,
        epochs: int,
        steps: int = 3,
        max_norm: float = 5.0,
        log: Optional[logging.Logger] = None,
        **_
    ) -> ControlMetric:
        self.epochs += 1
        self.model.train()
        sampler.set_epoch(epoch)
        start = time.time()
        debug = ControlMetric()
        with bar(desc=f'train-{self.device} (epoch-{epoch})', total=sampler.num_sens, leave=False, disable=(self.device != 0)) as pbar:
            for i, (inputs, masks, targets, sens) in enumerate(loader):
                loss, _debug = self.train_step(*to(self.device, inputs, masks, targets))
                (loss/steps).backward()
                if i % steps == 0 or (i+1) == len(loader):
                    optimizer.step()
                    self.clip(max_norm=max_norm)
                    optimizer.zero_grad()
                debug += _debug
                pbar.update(len(sens))
        debug.to(self.device)
        debug.sync()
        if log:
            elapsed = time.time()-start
            log.info(str(pbar))
            log.info(f'Epoch {epoch}/{epochs}: loss={debug.loss:.3f}, elapsed={elapsed:.2f} [{sampler.num_tokens/elapsed:.2f} tokens/s {sampler.num_sens/elapsed:.2f} sens/s]')
        return debug

    def clip(self, max_norm: float):
        self.model.clip_grad_norm_(max_norm=max_norm, norm_type=2)
        
    def train(self, *args, **kwargs):
        torch.cuda.set_device(self.device)
        if self.device == 0:
            output_folder = kwargs['output_folder']
            log = logger('train', path=f'{output_folder}/train.log', level=logging.DEBUG, dist=True)
        else:
            log = False
        return super().train(*args, **kwargs, log=log)
        
    def evaluate(
        self,
        data: Union[str, Dataset],
        output_folder: Optional[str] = None,
        batch_size: int = 100,
        log: Union[bool, logging.Logger] = False,
        **_
    ) -> Metric:
        dist.barrier()
        if isinstance(log, bool) and self.device == 0:
            log = logger('eval', path=f'{output_folder}/eval.log' if output_folder else None, level=logging.DEBUG, dist=True)
        metric = super().evaluate(data, batch_size=batch_size, log=log)
        metric.control.to(self.device).sync()
        metric.to(self.device).sync()
        if output_folder and self.device == 0:
            metric.save(f'{output_folder}/{data.name}.mt')
        return metric 
            
    def predict(
        self,
        data: Union[str, Dataset],
        output_folder: Optional[str] = None,
        batch_size: int = 100,
        log: Union[bool, logging.Logger] = False,
        **_
    ) -> Dataset:
        data = self.transform_data(data)
        if isinstance(log, bool) and self.device == 0:
            log = logger('predict', path=f'{output_folder}/predict.log' if output_folder else None, level=logging.DEBUG, dist=True)
        preds = super().predict(data, output_folder=None, batch_size=batch_size, log=log).sens
        pred = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(pred, preds)
        pred = data.__class__(flatten(pred), None).sort()
        if output_folder and self.device == 0:
            pred.save(f'{output_folder}/{filename(data.path)}')
        return pred