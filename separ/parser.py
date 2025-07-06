from __future__ import annotations
from torch.optim import AdamW, Optimizer
import torch, os, logging, time, wandb, shutil
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Sampler
from torch import nn 
from torch.distributed.tensor import distribute_tensor
import torch.distributed as dist 

from separ.model import Model 
from separ.utils import Config, to, ControlMetric, Metric, logger, bar, filename, save , is_distributed, WORLD_SIZE, flatten, is_main
from separ.data import InputTokenizer, TargetTokenizer, Dataset, Sentence

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["WANDB_SILENT"] = "true"

class Parser:
    OPTIMIZER = AdamW
    MODEL = Model
    METRIC = Metric
    PARAMS: list[str] = []
    DATASET: list[type]
    
    def __init__(
        self, 
        input_tkzs: list[InputTokenizer],
        target_tkzs: list[TargetTokenizer],
        model_confs: list[Config],
        device: int
    ):
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.model_confs = model_confs 
        for tkz in input_tkzs + target_tkzs:
            self.__setattr__(tkz.name, tkz)
        self.model = self.MODEL(*model_confs).to(device)
        if is_distributed():
            self.model.shard()
        self.epochs = 1
        self.device = device
        self.no_improv = [0, 0]
            
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({", ".join(f"{param}={getattr(self, param)}" for param in self.PARAMS)})'
        
    @property
    def INPUT_FIELDS(self) -> list[str]:
        return [tkz.field for tkz in self.input_tkzs]
    
    @property
    def TARGET_FIELDS(self) -> list[str]:
        return [tkz.field for tkz in self.target_tkzs]
    
    def barrier(self):
        if is_distributed():
            dist.barrier(device_ids=[self.device])

    @classmethod
    def load_data(cls, path: str) -> Dataset:
        """Loads a dataset from the list of supported datasets (DATASET).

        Args:
            path (str): Path to dataset.

        Returns:
            Dataset: Loaded dataset.
        """
        for c in cls.DATASET:
            if path.endswith(c.EXTENSION):
                data = c.from_file(path)
        return data 
    
    def transform_data(self, data: str | Dataset) -> Dataset:
        """Loads and transforms data from the list of supported datasets (DATASET).

        Args:
            data (Union[str, Dataset]): Path to dataset or dataset

        Returns:
            Dataset: Loaded and transformed dataset.
        """
        if isinstance(data, str):
            data = self.load_data(data) 
        data.sens = list(bar(map(self.transform, data), desc=f'{self.NAME}[transform]', leave=False, total=len(data)))
        return data 
    
    def transform(self, sen: Sentence) -> Sentence:
        return sen 
    
    def loader(self, data: Dataset, batch_size: int, shuffle: bool) -> tuple[DataLoader, Sampler]:
        return data.loader(batch_size, shuffle=shuffle, collate=self.collate)
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser.add_argument('-c', '--conf', type=str, default=None, help='Configuration file')
        argparser.add_argument('-d', '--device', type=int, default=0, help='CUDA device')
        argparser.add_argument('--load', type=str, default=None, help='Load the parser')
        argparser.add_argument('--seed', type=int, default=123, help='Parser seed')
        return argparser
       
    def save(self, path: str):
        self.barrier()
        if is_distributed():
            sharded_sd = self.model.state_dict()
            state = {}
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if is_main():
                    state[param_name] = full_param.cpu()
                else:
                    del full_param
        else:
            state = self.model.state_dict()
        if is_main():
            model = self.model 
            self.__delattr__('model')
            self.state = state
            torch.save(self, path)
            self.__delattr__('state')
            self.model = model 
        self.barrier()
            
    @classmethod
    def load(cls, path: str, device: int) -> Parser:
        parser = torch.load(path, weights_only=False, map_location='cpu')
        parser.model = parser.MODEL(*parser.model_confs).to(device)
        parser.device = device 
        if is_distributed():
            parser.model.shard()
            parser.load_state(parser.state)
        else:
            parser.model.load_state_dict(parser.state)
        delattr(parser, 'state')
        return parser 
    
    def load_state(self, state: dict[str, torch.Tensor]):
        """Loads the state from a file"""
        self.barrier()
        if is_distributed():
            current_sharded = self.model.state_dict()
            new_sharded = dict()
            for param_name, full_tensor in state.items():
                sharded_param = current_sharded.get(param_name)
                sharded_tensor = distribute_tensor(full_tensor, sharded_param.device_mesh, sharded_param.placements)
                new_sharded[param_name] = nn.Parameter(sharded_tensor)
            self.model.load_state_dict(new_sharded, assign=True)
        else:
            self.model.load_state_dict(state)
            
    def train(
        self, 
        train: str | Dataset, 
        dev: str | Dataset,
        test: list[str | Dataset],
        output_folder: str,
        batch_size: int = 100,
        epochs: int = 100,
        lr: float = 1e-5,
        train_patience: int = 5,
        dev_patience: int = 10,
        run_name: str | None = None,
        **kwargs
    ):
        """Parser training.

        Args:
            train (str | Dataset): Training dataset.
            dev (str | Dataset): Validation dataset.
            test (list[str | Dataset]): Test datasets.
            output_folder (str): Path to store training results.
            batch_size (int): Batch-size.
            epochs (int): number of training epochs.
            lr (float): Learning rate.
            train_patience (int): Number of epochs with no training improvement.
            dev_patience (int): Number of epochs with no development improvement.
            load (str | None): Whether to load the state of a previous parser.pt.
        """
        
        train, dev, *test = map(self.transform_data, [train, dev, *test])
        if is_main():
            os.makedirs(output_folder, exist_ok=True)
            log = logger('train', path=f'{output_folder}/train.log', level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Saving training at {output_folder}\n' + '\n'.join(f'{name} = {str(value)}' for name, value in locals().items() if name != 'kwargs'))
            log.info(f'train: {train}')
            log.info(f'dev: {dev}')
            log.info(f'test: ' + ', '.join(map(repr, test)))
            run = wandb.init(project='trasepar', name=run_name, config={'server': os.uname()[1], 'parser': repr(self), 'data': train.path})
        else:
            log = None
        optimizer = self.OPTIMIZER(self.model.parameters(), lr=lr)
        loader, sampler = self.loader(train, batch_size=batch_size, shuffle=True)
        best_metric, best_loss = self.METRIC, torch.inf
        history = []
        for epoch in range(self.epochs, epochs+1):
            debug = self.epoch(epoch, optimizer, loader, sampler, epochs, log=log, **kwargs)

            # validation step 
            dev_metric = self.evaluate(data=dev, batch_size=batch_size, output_folder=output_folder, log=log)
            # check if validation improves 
            self.save(f'{output_folder}/parser.pt')
            if dev_metric.improves(best_metric):
                # all ranks should update best metrics
                best_metric = dev_metric 
                self.no_improv[1] = 0
                shutil.copy(f'{output_folder}/parser.pt', f'{output_folder}/best.pt')
                if log:
                    log.debug('(improved)')
            else:
                self.no_improv[1] += 1
            history.append((debug, dev_metric))
            if log:
                run.log({'train/loss': debug.loss} | {f'dev/{metric}': value for metric, value in dev_metric.items(scale=100)})
                
            if best_loss > debug.loss:
                best_loss = debug.loss 
                self.no_improv[0] = 0
            else:
                self.no_improv[0] +=1
                
            if self.no_improv[0] == train_patience:
                if log:
                    log.warning('No more improvement in the train set')
                break
            if self.no_improv[1] == dev_patience:
                if log:
                    log.warning('No more improvement in the dev set')
                break 
            if debug.loss < 1e-12:
                if log:
                    log.warning('Zero loss reached')
                break 

        torch.cuda.empty_cache()
        self.load_state(torch.load(f'{output_folder}/best.pt', map_location='cpu', weights_only=False).state)
        self.trained = True 
        self.save(f'{output_folder}/parser.pt')
        if is_main():
            os.remove(f'{output_folder}/best.pt')
        
        # predict test sets
        for _test in test:
            self.evaluate(_test, output_folder=output_folder, batch_size=batch_size, log=log)
            self.predict(_test, output_folder=output_folder, batch_size=batch_size, log=log)
            
        if log:
            save(history, f'{output_folder}/history.pickle')
            run.finish()

    def epoch(
        self, 
        epoch: int,
        optimizer: Optimizer,
        loader: DataLoader,
        sampler: Sampler,
        epochs: int,
        steps: int = 3,
        max_norm: float = 5.0,
        log: logging.Logger | None = None,
        **_
    ) -> ControlMetric:
        self.barrier()
        self.epochs += 1
        self.model.train()
        sampler.set_epoch(epoch)
        start = time.time()
        debug = ControlMetric()
        with bar(desc=f'train (epoch-{epoch})', total=sampler.num_sens, leave=False, disable=not is_main()) as pbar:
            for i, (inputs, masks, targets, sens) in enumerate(loader):
                loss, _debug = self.train_step(*to(self.device, inputs, masks, targets))
                if loss.isnan():
                    continue 
                (loss/steps).backward()
                if (i+1) % steps == 0 or (i+1) == len(loader):
                    optimizer.step()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
                    optimizer.zero_grad()
                debug += _debug
                pbar.update(len(sens))
                pbar.set_postfix_str(repr(debug))
                self.barrier()
        if log:
            elapsed = time.time()-start
            log.info(str(pbar))
            log.info(f'Epoch {epoch}/{epochs}: loss={debug.loss:.3f}, elapsed={elapsed:.2f} [{sampler.num_tokens/elapsed:.2f} tokens/s {sampler.num_sens/elapsed:.2f} sens/s]')
        return debug.to(self.device).sync()
    
            
    def evaluate(
        self,
        data: str | Dataset,
        output_folder: str | None = None,
        batch_size: int = 100,
        log: logging.Logger | None = None,
        **_
    ) -> Metric:
        """Parser distributed evaluation.

        Args:
            data (str | Dataset): Input dataset.
            conf (Config): Evaluation configuration (device, batch-size, output folder).
            output_folder (Optional[str]): Path to store the evaluation metric.
            batch_size (int): Inference batch size.
            log (logging.Logger | None): Logger.

        Returns:
            Metric: Evaluation metric.
        """
        self.barrier()
        if log is None and is_main():
            log = logger('eval', path=f'{output_folder}/eval.log' if output_folder else None, level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Evaluation from {data}')
        data = self.transform_data(data)
        self.model.eval()
        start = time.time()
        loader, sampler = self.loader(data, batch_size, shuffle=True)
        control, metric = ControlMetric(), self.METRIC
        with bar(desc=f'eval', total=sampler.num_sens, leave=False, disable=not is_main()) as pbar:
            for *batch, sens in loader:
                _control, _metric = self.eval_step(*to(self.device, *batch), sens)
                control += _control 
                metric += _metric 
                pbar.update(len(sens))
                pbar.set_postfix_str(repr(control))
                self.barrier()
        control.elapsed = time.time()-start
        metric.control = control.to(self.device).sync()
        metric.to(self.device).sync()
        if log:
            log.info(f'{data.name}: {metric} [{control} {data.n_tokens/control.elapsed:.2f} tokens/s {len(data)/control.elapsed:.2f} sens/s]')
            if output_folder and is_main():
                metric.save(f'{output_folder}/{data.name}.mt')
        return metric
    
    def predict(
        self,
        data: str | Dataset,
        output_folder: str | None = None,
        batch_size: int = 100,
        log: logging.Logger | None = None,
        **_
    ) -> Dataset:
        """Parser distributed prediction.

        Args:
            data (str | Dataset): Input dataset.
            conf (Config): Evaluation configuration (device, batch-size, output folder).
            output_folder (Optional[str]): Path to store the evaluation metric.
            batch_size (int): Inference batch size.
            log (logging.Logger | None): Logger.

        Returns:
            Dataset: Predicted dataset.
        """
        self.barrier()
        if log is None and is_main():
            log = logger('predict', path=f'{output_folder}/predict.log' if output_folder else None, level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Prediction from {data}')
        data = self.transform_data(data)
        loader, sampler = self.loader(data, batch_size=batch_size, shuffle=True)
        start = time.time()
        self.model.eval()
        preds = []
        with bar(desc='pred', total=sampler.num_sens, leave=False, disable=not is_main()) as pbar:
            for inputs, masks, _, sens in loader:
                preds += self.pred_step(*to(self.device, inputs, masks), sens)
                pbar.update(len(sens))
                self.barrier()
        if is_distributed():
            pred = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(pred, preds)
            pred = data.__class__(flatten(pred), None).sort()
        else:
            pred = data.__class__(preds, None).sort()
        if log:
            elapsed = time.time()-start
            log.info(f'{data.name}: {data.n_tokens/elapsed:.2f} tokens/s {len(data)/elapsed:.2f} sens/s]')
            if output_folder and is_main():
                pred.save(f'{output_folder}/{filename(data.path)}')
        return pred
                
    def train_step(
        self,
        inputs: list[torch.Tensor], 
        masks: list[torch.Tensor],
        targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ControlMetric]:
        raise NotImplementedError
            
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        sens: list[Sentence]
    ) -> tuple[ControlMetric, Metric]:
        raise NotImplementedError
        
    @torch.no_grad()
    def pred_step(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor],
        sens: list[Sentence]
    ) -> list[Sentence]:
        raise NotImplementedError
    

        
        
        
