from __future__ import annotations
from typing import List, Union, Tuple, Optional
from torch.optim import AdamW, Optimizer
import torch, os, logging, time, wandb 
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Sampler
from torch import nn 

from separ.model import Model 
from separ.utils import Config, to, ControlMetric, Metric, logger, bar, filename, save 
from separ.data import InputTokenizer, TargetTokenizer, Dataset, Sentence

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["WANDB_SILENT"] = "true"

class Parser:
    OPTIMIZER = AdamW
    MODEL = Model
    METRIC = Metric
    PARAMS: List[str] = []
    DATASET: List[type]
    
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
        self.model = self.MODEL(*model_confs).to(device)
        self.epochs = 1
        self.device = device
            
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({", ".join(f"{param}={getattr(self, param)}" for param in self.PARAMS)})'
        
    @property
    def INPUT_FIELDS(self) -> List[str]:
        return [tkz.field for tkz in self.input_tkzs]
    
    @property
    def TARGET_FIELDS(self) -> List[str]:
        return [tkz.field for tkz in self.target_tkzs]
    
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
    
    def transform_data(self, data: Union[str, Dataset]) -> Dataset:
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
    
    def loader(self, data: Dataset, batch_size: int, shuffle: bool) -> Tuple[DataLoader, Sampler]:
        return data.loader(batch_size, shuffle=shuffle, collate=self.collate)
    
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser.add_argument('-c', '--conf', type=str, default=None, help='Configuration file')
        argparser.add_argument('-d', '--device', type=int, default=0, help='CUDA device')
        argparser.add_argument('--load', type=str, default=None, help='Load the parser')
        argparser.add_argument('--seed', type=int, default=123, help='Parser seed')
        return argparser
       
    def save(self, path: str):
        model = self.model 
        self.__delattr__('model')
        self.state = model.state_dict()
        torch.save(self, path)
        self.__delattr__('state')
        self.model = model 
            
    @classmethod
    def load(cls, path: str, device) -> Parser:
        # this parser does not have the model, only the state 
        parser = torch.load(path, weights_only=False, map_location='cpu')
        # build the model and load the state
        parser.model = parser.MODEL(*parser.model_confs).to(device)
        parser.model.load_state_dict(parser.state)
        delattr(parser, 'state')
        parser.device = device
        return parser 
    
    def load_state(self, path: str):
        """Loads the state from a file"""
        self.model.load_state_dict(torch.load(path, weights_only=False, map_location=torch.device(self.device)).state)
            
    def train(
        self, 
        train: Union[str, Dataset], 
        dev: Union[str, Dataset],
        test: List[Union[str, Dataset]],
        output_folder: str,
        batch_size: int = 100,
        epochs: int = 100,
        lr: float = 1e-5,
        log: Union[bool, logging.Logger] = True, 
        train_patience: int = 5,
        dev_patience: int = 10,
        run_name: Optional[str] = None,
        **kwargs
    ):
        """Parser training.

        Args:
            train (Union[str, Dataset]): Training dataset.
            dev (Union[str, Dataset]): Validation dataset.
            test (List[Union[str, Dataset]]): Test datasets.
            output_folder (str): Path to store training results.
            batch_size (int): Batch-size.
            epochs (int): number of training epochs.
            lr (float): Learning rate.
            log (Union[bool, logging.Logger]): Logger to display and store logs.
            train_patience (int): Number of epochs with no training improvement.
            dev_patience (int): Number of epochs with no development improvement.
            load (Optional[str]): Whether to load the state of a previous parser.pt.
        """
        
        train, dev, *test = map(self.transform_data, [train, dev, *test])
        if isinstance(log, bool) and log:
            os.makedirs(output_folder, exist_ok=True)
            log = logger('train', path=f'{output_folder}/train.log', level=logging.DEBUG)
        if log:
            log.info(self)
            log.info(self.model)
            log.info(f'Saving training at {output_folder}\n' + '\n'.join(f'{name} = {str(value)}' for name, value in locals().items() if name != 'kwargs'))
            log.info(f'train: {train}')
            log.info(f'dev: {dev}')
            log.info(f'test: ' + ', '.join(map(repr, test)))
            run = wandb.init(project='trasepar', name=run_name, config={'server': os.uname()[1], 'parser': repr(self), 'data': train.path})
        optimizer = self.OPTIMIZER(self.model.parameters(), lr=lr)
        loader, sampler = self.loader(train, batch_size=batch_size, shuffle=True)
        best_metric, best_loss, train_improv, dev_improv = self.METRIC, torch.inf, train_patience, dev_patience 
        history = []
        for epoch in range(self.epochs, epochs+1):
            debug = self.epoch(epoch, optimizer, loader, sampler, epochs, log=log, **kwargs)

            # validation step 
            dev_metric = self.evaluate(data=dev, batch_size=batch_size, output_folder=output_folder, log=log)
            # check if validation improves 
            if dev_metric.improves(best_metric):
                # all ranks should update best metrics
                best_metric = dev_metric 
                dev_improv = dev_patience
                self.save(f'{output_folder}/parser.pt')
                if log:
                    log.debug('(improved)')
            else:
                dev_improv -= 1
            history.append((debug, dev_metric))
            if log:
                run.log({'train/loss': debug.loss} | {f'dev/{metric}': value for metric, value in dev_metric.items(scale=100)})
                
            if best_loss > debug.loss:
                best_loss = debug.loss 
                train_improv = train_patience
            else:
                train_improv -= 1 
                
            if train_improv == 0:
                if log:
                    log.warning('No more improvement in the train set')
                break
            if dev_improv == 0:
                if log:
                    log.warning('No more improvement in the dev set')
                break 
            if debug.loss < 1e-12:
                if log:
                    log.warning('Zero loss reached')
                break 
        
        self.trained = True 
        torch.cuda.empty_cache()
        self.load_state(f'{output_folder}/parser.pt')
        
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
        log: Optional[logging.Logger] = None,
        **_
    ) -> ControlMetric:
        self.epochs += 1
        self.model.train()
        sampler.set_epoch(epoch)
        start = time.time()
        debug = ControlMetric()
        with bar(desc=f'train (epoch-{epoch})', total=sampler.num_sens, leave=False) as pbar:
            for i, (inputs, masks, targets, sens) in enumerate(loader):
                loss, _debug = self.train_step(*to(self.device, inputs, masks, targets))
                (loss/steps).backward()
                if i % steps == 0 or (i+1) == len(loader):
                    optimizer.step()
                    self.clip(max_norm=max_norm)
                    optimizer.zero_grad()
                debug += _debug
                pbar.update(len(sens))
        if log:
            elapsed = time.time()-start
            log.info(str(pbar))
            log.info(f'Epoch {epoch}/{epochs}: loss={debug.loss:.3f}, elapsed={elapsed:.2f} [{sampler.num_tokens/elapsed:.2f} tokens/s {sampler.num_sens/elapsed:.2f} sens/s]')
        return debug
    
    def clip(self, max_norm: float):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
            
    def evaluate(
        self,
        data: Union[str, Dataset],
        output_folder: Optional[str] = None,
        batch_size: int = 100,
        log: Union[bool, logging.Logger] = True,
        **_
    ) -> Metric:
        """Parser distributed evaluation.

        Args:
            data (Union[str, Dataset]): Input dataset.
            conf (Config): Evaluation configuration (device, batch-size, output folder).
            output_folder (Optional[str]): Path to store the evaluation metric.
            batch_size (int): Inference batch size.
            log (Union[bool, logging.Logger]): Logger. If True, create a new logger. If False, do not display logs.

        Returns:
            Metric: Evaluation metric.
        """
        if isinstance(log, bool) and log:
            log = logger('eval', path=f'{output_folder}/eval.log' if output_folder else None, level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Evaluation from {data}')
        data = self.transform_data(data)
        self.model.eval()
        start = time.time()
        loader, sampler = self.loader(data, batch_size, shuffle=False)
        control, metric = ControlMetric(), self.METRIC
        if log:
            pbar = bar(desc=f'eval', total=sampler.num_sens, leave=False)
        for *batch, sens in loader:
            _control, _metric = self.eval_step(*to(self.device, *batch), sens)
            control += _control 
            metric += _metric 
            if log:
                pbar.update(len(sens))
                pbar.set_postfix_str(repr(control))
        control.elapsed = time.time()-start
        metric.control = control 
        if log:
            pbar.close()
            log.info(f'{data.name}: {metric} [{control} {data.n_tokens/control.elapsed:.2f} tokens/s {len(data)/control.elapsed:.2f} sens/s]')
            if output_folder:
                metric.save(f'{output_folder}/{data.name}.mt')
        return metric
    
    def predict(
        self,
        data: Union[str, Dataset],
        output_folder: Optional[str] = None,
        batch_size: int = 100,
        log: Union[bool, logging.Logger] = True,
        **_
    ) -> Dataset:
        """Parser distributed prediction.

        Args:
            data (Union[str, Dataset]): Input dataset.
            conf (Config): Evaluation configuration (device, batch-size, output folder).
            output_folder (Optional[str]): Path to store the evaluation metric.
            batch_size (int): Inference batch size.
            log (Union[bool, logging.Logger]): Logger. If True, create a new logger. If False, do not display logs.

        Returns:
            Dataset: Predicted dataset.
        """
        if isinstance(log, bool) and log:
            log = logger('predict', path=f'{output_folder}/predict.log' if output_folder else None, level=logging.DEBUG)
            log.info(self)
            log.info(self.model)
            log.info(f'Prediction from {data}')
        data = self.transform_data(data)
        loader, sampler = self.loader(data, batch_size=batch_size, shuffle=False)
        start = time.time()
        self.model.eval()
        preds= []
        if log:
            pbar = bar(desc='pred', total=sampler.num_sens, leave=False)
        for inputs, masks, _, sens in loader:
            preds += self.pred_step(*to(self.device, inputs, masks), sens)
            if log:
                pbar.update(len(sens))
        pred = data.__class__(preds, None).sort()
        if log:
            pbar.close()
            elapsed = time.time()-start
            log.info(f'{data.name}: {data.n_tokens/elapsed:.2f} tokens/s {len(data)/elapsed:.2f} sens/s]')
            if output_folder:
                pred.save(f'{output_folder}/{filename(data.path)}')
        return pred
                
    def train_step(
        self,
        inputs: List[torch.Tensor], 
        masks: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ControlMetric]:
        raise NotImplementedError
            
    @torch.no_grad()
    def eval_step(
        self, 
        inputs: List[torch.Tensor],
        masks: List[torch.Tensor],
        targets: List[torch.Tensor],
        sens: List[Sentence]
    ) -> Tuple[ControlMetric, Metric]:
        raise NotImplementedError
        
    @torch.no_grad()
    def pred_step(
        self,
        inputs: List[torch.Tensor],
        masks: List[torch.Tensor],
        sens: List[Sentence]
    ) -> List[Sentence]:
        raise NotImplementedError
    

        
        
        
