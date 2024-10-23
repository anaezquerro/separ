from torch import nn 
from typing import List, Tuple, Union, Optional, Dict, Callable
import os, shutil, torch, pickle, time, logging 
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np 
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
from carbontracker.tracker import CarbonTracker

from separ.utils import to, bar, logger, init_folder, History, Epoch, Metric, merge, fdict, scale_dict
from separ.utils.metric import Metric
from separ.data import SDP, CoNLL, PTB
from separ.structs import AbstractDataset
from separ.model import Model

# torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Parser:
    """General implementation of a neural parser.
    
    Global variables:
        NAME (str): Name of the parser.
        MODEL (Callable): Callable to initialize the neural Model..
        METRIC (Callable): Callable to initialize and evaluation metric.
        DATASET (Callable): Callable to initialize an AbstractDataset.
        PARAMS (List[str]): List of initialization hyperparameters.
        OPTIMIZER (Callable): Callable to PyTorch optimizer.
        
    Methods:
        train
        evaluate
        predict
        control
        reference
        load
        save
    """
    NAME: str
    MODEL: Callable
    METRIC: Callable
    DATASET: Callable
    PARAMS: List[str] 
    OPTIMIZER: Callable = AdamW
    
    def __init__(self, model: Model, input_tkzs, target_tkzs, device: Union[str, int]):
        """Shared initialization between all parsers.

        Args:
            model (Model): Neural model.
            input_tkzs (List[AbstractTokenizer]): Input tokenizers.
            target_tkzs (List[AbstractTokenizer]): Target tokenizers.
            device (Union[str, int]): CUDA device.
        """
        self.model = model
        self.confs = model.confs 
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.device = device if isinstance(device, str) else f'cuda:{device}'
        for tkz in self.input_tkzs + self.target_tkzs:
            self.__setattr__(tkz.name, tkz)
        self.trained = False # flag to ensure that the parser has finished a training cycle
            
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + \
            ', '.join(f'{param}={getattr(self, param)}' for param in self.PARAMS) + \
                ')'
                
    def train(
        self,
        train: Union[str, AbstractDataset],
        dev: Union[str, AbstractDataset],
        test: List[Union[str, AbstractDataset]], 
        path: str, 
        epochs: int, 
        batch_size: int,
        lr: float,
        max_norm: float = 2.0,
        train_patience: int = 5,
        dev_patience: int = 10,
        num_workers: int = 1,
        ref: bool = False,
        **_
    ):
        """Training cycle.

        Args:
            train (Union[str, AbstractDataset]): _description_
            dev (Union[str, AbstractDataset]): _description_
            test (List[Union[str, AbstractDataset]]): _description_
            path (str): _description_
            epochs (int, optional): _description_. Defaults to 500.
            batch_size (int, optional): _description_. Defaults to 100.
            lr (float, optional): _description_. Defaults to 1e-4.
            max_norm (float, optional): _description_. Defaults to 2.0.
            train_patience (int, optional): _description_. Defaults to 5.
            dev_patience (int, optional): _description_. Defaults to 10.
            num_workers (int, optional): _description_. Defaults to 1.
            ref (bool, optional): _description_. Defaults to False.
        """
        init_folder(path)
        log = logger('train', path=f'{path}/train.log', level=logging.DEBUG)
        log.info(self)
        log.info(self.model)
        log.info(f'Saving training at {path}')
        self.save(f'{path}/parser.pt')
        self.optimizer = self.OPTIMIZER(self.model.parameters(), lr=lr)
        
        # prepare data 
        train, dev, *test = map(lambda x: self.build_data(x, num_workers), [train, dev, *test])
        log.info(f'train: {train}')
        log.info(f'dev: {dev}')
        log.info(f'test: ' + ', '.join(map(repr, test)))
        train_dl = train.loader(self.collate, batch_size=batch_size, pred=True)
        best_metric, best_loss, train_improv, dev_improv = self.METRIC(), np.Inf, train_patience, dev_patience
        history = History()
        if ref:
            self.reference(dev, batch_size, num_workers=num_workers, log=log)
        self.trained = False 
        for i in range(epochs):
            # apply epoch
            epoch, loss, elapsed = self.epoch(Epoch(i+1), train, train_dl, log, max_norm)
            log.info(f'Epoch {i+1}/{epochs}: loss={loss:.3f}, {train.n_tokens/elapsed:.2f} token/s, {len(train)/elapsed:.2f} sent/s, {elapsed:.2f}s elapsed')
            
            # evaluation step 
            dev_control, dev_metric = self.control(dev, batch_size, num_workers=num_workers, log=log)
            epoch.add_subset(dev.name, **dev_control, metric=dev_metric)
            if dev_metric.improves(best_metric):
                log.debug('(improved)')
                best_metric = dev_metric
                self.save(f'{path}/parser.pt')
                dev_improv = dev_patience 
                for _test in test:
                    test_control, test_metric = self.control(
                        _test, batch_size, num_workers=num_workers, log=log, path=path 
                    )
                    epoch.add_subset(_test.name, **test_control, metric=test_metric)
                history.add(epoch, best=True)
                dev_metric.add_control(**dev_control)
                dev_metric.save(f'{path}/{dev.name}.pickle')
            else:
                dev_improv -= 1 
                history.add(epoch, best=False)
                
            if best_loss > loss:
                best_loss = loss 
                train_improv = train_patience 
            else:
                train_improv -= 1
            
            if train_improv == 0:
                log.warn('No more improvement in the train set')
                break
            if dev_improv == 0:
                log.warn('No more improvement in the dev set')
                break 
            if loss < 1e-12:
                log.warn('Zero loss reached')
                break 
                
        # load model weights and save test prediction and metrics
        torch.cuda.empty_cache()
        self = self.load(f'{path}/parser.pt', self.device)
        self.trained = True 
        self.save(f'{path}/parser.pt')
        for _test in test + [dev]:
            try:
                getattr(history.best, _test.name)
            except AttributeError:
                continue 
            else:
                metric = getattr(history.best, _test.name)['metric']
                # update with speed metrics and save predictions
                _, elapsed = self.predict(_test, path=f'{path}/{_test.name}.{_test.EXTENSION}', batch_size=batch_size, num_workers=num_workers)
                metric.add_control(**{'token/s': _test.n_tokens/elapsed, 'sent/s': len(_test)/elapsed, 'elapsed': elapsed})
                metric.save(f'{path}/{_test.name}.pickle')
        history.save(f'{path}/history.pickle')
        
    def epoch(
        self, 
        epoch: Epoch,
        train: AbstractDataset, 
        train_dl: DataLoader, 
        log: Optional[logging.Logger], 
        max_norm: float
    ) -> Tuple[Epoch, float, float]:
        """Compute a training epoch.

        Args:
            epoch (Epoch): Epoch object to track training.
            train (AbstractDataset): Train dataset.
            train_dl (DataLoader): Train data loader.
            sampler (TokenizedSampler): Training batch sampler.
            log (logging.Logger): Logging.
            max_norm (float): Clip norm value.

        Returns:
            Tuple[Epoch, float, float]: Update epoch, loss and elapsed time.
        """
        torch.cuda.empty_cache()
        train_dl.batch_sampler.step()
        self.model.train()
        loss, elapsed = 0, 0
        with bar(total=len(train), desc='train', leave=True) as pbar:
            for inputs, targets, masks, sens in train_dl:
                # forward pass 
                inputs, targets, masks = to(self.device, inputs, targets, masks)
                start = time.time()
                _loss, debug = self.train_step(inputs, targets, masks)
                end = time.time()
                
                # backward pass 
                self.optimizer.zero_grad()
                _loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
                self.optimizer.step()
                
                # update pbar 
                pbar.update(len(sens))
                pbar.set_postfix_str(f'loss={_loss.item():.3f}, {sum(map(len, sens))/(end-start):.2f} token/s, {len(sens)/(end-start):.2f} sent/s {fdict(debug, n=2)}')
                
                # update variables 
                loss += _loss.item()
                elapsed += (end - start)
        pbar = str(pbar).split('|')
        if log:
            log.info(pbar[0] + pbar[1][:130] + pbar[2])
        epoch.add_subset(train.name, **debug)
        return epoch, loss/len(train_dl), elapsed
        
    def evaluate(
        self,
        data: Union[AbstractDataset, str],
        batch_size: int,
        path: Optional[str] = None,
        num_workers: int = 1,
        log: Optional[logging.Logger] = None, 
    ) -> Metric:
        """Evaluation.

        Args:
            data (Union[AbstractDataset, str]): Input dataset to evaluate.
            batch_size (int, optional): Batch size. Defaults to 100.
            path (Optional[str]): Folder path to store the evaluation metric.
            num_workers (int, optional): Number of workers to parallelize execution. Defaults to 1.
            log (Optional[logging.Logger], optional): Python logger. Defaults to None.

        Returns:
            Metric: Evaluation metric.
        """
        self.model.eval()
        data = self.build_data(data, num_workers)
        if log is None:
            log = logger('eval', level=logging.DEBUG, path=f'{path}/eval-{data.name}.log' if path is not None else None)
            log.info(self)
            log.info(self.model)
        log.info(f'Evaluating {self} in {data.path}' + (f' -> {path}/{data.name}.pickle' if path else ''))
        loader = data.loader(self.collate, batch_size=batch_size, pred=True)
        preds, elapsed = [], 0 
        with bar(total=len(data), desc=f'eval:{data.name}', leave=False) as pbar:
            for inputs, _, masks, sents in loader:
                # prediction step 
                batch = to(self.device, inputs, masks)
                start = time.time()
                preds += self.pred_step(*batch, sents)
                end = time.time()
                
                # update variables
                pbar.update(len(sents))
                elapsed += (end-start)
        metric = self.METRIC(data.__class__(preds, None), data, num_workers=num_workers)
        log.info(f'{data.name}: {metric} [{data.n_tokens/elapsed:.2f} token/s, {len(data)/elapsed:.2f} sent/s, {elapsed:.2f}s elapsed]')
        if path:
            metric.add_control(**{'token/s': data.n_tokens/elapsed, 'sent/s': len(data)/elapsed, 'elapsed': elapsed})
            metric.save(f'{path}/{data.name}.pickle')
        torch.cuda.empty_cache()
        return metric

    def predict(
        self, 
        data: Union[AbstractDataset, str],
        batch_size: int,
        path: Optional[str] = None,
        num_workers: int = 1,
        log: Optional[logging.Logger] = None
    ) -> Tuple[AbstractDataset, float]:
        """Prediction step.

        Args:
            data (Union[AbstractDataset, str]): Input dataset.
            batch_size (int): Batch size.
            path (str): Path to store the output dataset.
            num_workers (int): Parallelization threads.
            log (logging.Logger): Python logger.

        Returns:
            Tuple[AbstractDataset, float]: Predicted dataset and elapsed time.
        """
        data = self.build_data(data, num_workers)
        if log is None:
            log = logger('predict', level=logging.DEBUG, path='.'.join(path.split('.')[:-1]) + 'log' if path is not None else None)
            log.info(self)
            log.info(self.model)
        self.model.eval()
        log.info(f'Prediction with {self} in {data.path}' + (f' -> {path}' if path else ''))
        loader = data.loader(self.collate, batch_size=batch_size, pred=True)
        preds, elapsed = [], 0
        with bar(total=len(data), desc=f'predict:{data.name}', leave=False) as pbar:
            for inputs, _, masks, sents in loader:
                batch = to(self.device, inputs, masks)
                start = time.time()
                preds += self.pred_step(*batch, sents) 
                elapsed += (time.time() - start)
                pbar.update(len(sents))
        log.info(f'{data.name}: {data.n_tokens/elapsed:.2f} token/s, {len(data)/elapsed:.2f} sent/s, {elapsed:.2f}s elapsed')
        pred = data.__class__(preds, path or data.path)
        if path:
            pred.save(path)
        torch.cuda.empty_cache()
        return pred, elapsed 
    
    def control(
        self,
        data: Union[AbstractDataset, str],
        batch_size: int,
        path: Optional[str] = None,
        num_workers: int = 1,
        log: logging.Logger =  logger('debug', level=logging.DEBUG),
    ) -> Tuple[Dict[str, float], Metric]:
        """Performs a debugging step to recover evaluation and control metrics.

        Args:
            data (Union[AbstractDataset, str]): Input dataset.
            batch_size (int, optional): Batch size. Defaults to 100.
            path (Optional[str]): Folder path to store the prediction and control metrics.
            num_workers (int, optional): Number of workers. Defaults to 1.
            log (Optional[logging.Logger], optional): Python logger. Defaults to None.

        Returns:
            Tuple[Dict[str, float], Metric]: Control metrics and evaluation metric.
        """
        self.model.eval()
        data = self.build_data(data, num_workers)
        loader = data.loader(self.collate, batch_size=batch_size, pred=True)
        preds, elapsed, control = [], 0, dict()
        with bar(total=len(data), desc=f'control:{data.name}', leave=False) as pbar:
            for *batch, sents in loader:
                batch = to(self.device, *batch)
                start = time.time()
                _control, _preds = self.control_step(*batch, sents)
                elapsed += (time.time()-start)
                preds += _preds
                control = merge(control, _control)
                pbar.update(len(sents))
        pred = data.__class__(preds, data.name)
        metric = self.METRIC(pred, data, num_workers=num_workers)
        control = scale_dict(control, 1/len(loader))
        log.info(f'{data.name}: {metric} [{data.n_tokens/elapsed:.2f} token/s, {len(data)/elapsed:.2f} sent/s, {elapsed:.2f}s elapsed]')
        log.info(f'[info]: {fdict(control, n=2)}')
        if path:
            pred.save(f'{path}/{data.name}.{data.EXTENSION}')
            metric.add_control(**control)
            metric.save(f'{path}/{data.name}.pickle')
        return control, metric 
    
    def reference(
        self,
        data: Union[AbstractDataset, str],
        batch_size: int,
        num_workers: int = 1,
        log: logging.Logger = logger('control', level=logging.DEBUG)
    ) -> Metric:
        """Control step (it compares that prediction with gold annotations).

        Args:
            data (Union[AbstractDataset, str]): Input dataset.
            batch_size (int, optional): Batch size. Defaults to 100.
            num_workers (int, optional): Number of workers. Defaults to 1.
            log (Optional[logging.Logger], optional): Python logger. Defaults to None.

        Returns:
            Metric: Evaluation metric.
        """
        self.model.eval()
        data = self.build_data(data, num_workers)
        loader = data.loader(self.collate, batch_size=batch_size, pred=True)
        preds, elapsed = [], 0
        with bar(total=len(data), desc=f'ref:{data.name}', leave=False) as pbar:
            for _, targets, masks, sents in loader:
                # forward step
                batch = to(self.device, targets, masks)
                start = time.time()
                preds += self.ref_step(*batch, sents)
                end = time.time()
                
                # update 
                pbar.update(len(sents))
                elapsed += (end-start)
        metric = self.METRIC(data.__class__(preds, None), data, num_workers=num_workers)
        log.info(f'{data.name}: {metric} [{data.n_tokens/elapsed:.2f} token/s, {len(data)/elapsed:.2f} sent/s, {elapsed:.2f}s elapsed]')
        return metric 
    
        
    def save(self, path: str):
        model = self.model
        self.state = model.state_dict()
        self.__delattr__('model')
        torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.model = model 
        
    @classmethod
    def load(cls, path: str, device: str = 'cuda:0'):
        parser = torch.load(path, map_location=device)
        parser.model = parser.MODEL(*parser.confs).to(device)
        parser.model.load_state_dict(parser.state)
        delattr(parser, 'state')
        parser.device = device 
        return parser 
        
    @classmethod
    def add_arguments(cls, argparser: ArgumentParser):
        argparser.add_argument('-p', '--path', type=str, help='Model path')
        argparser.add_argument('-c', '--conf', type=str, help='Configuration file')
        argparser.add_argument('-d', '--data', type=str, default=None, help='Building data')
        argparser.add_argument('--device', type=int, default=0, help='CUDA device')
        argparser.add_argument('--pretest', action='store_true', help='Test encodings before parser building')
        argparser.add_argument('--load', action='store_true', help='Load the parser from the configuration file')
        return argparser
    
    def build_data(self, data: Union[str, AbstractDataset], num_workers: int = 1) -> AbstractDataset:
        if isinstance(data, str):
            data = self.__class__.load_data(data, num_workers=num_workers)
        if all(sent._transformed for sent in data):
            return data
        if num_workers > 1:
            batch_size = int(len(data)/num_workers + 0.5)
            args = [(data[i:i+batch_size], *self.TRANSFORM_ARGS) for i in range(0, len(data), batch_size)]
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                _ = list(pool.map(self.__class__.transform_data, *zip(*args)))
        else:
            self.__class__.transform_data(data, *self.TRANSFORM_ARGS)
        return data
    
    @classmethod
    def load_data(cls, path: str, num_workers: int) -> Union[CoNLL, PTB, SDP]:
        if isinstance(cls.DATASET, list):
            for b in cls.DATASET:
                if path.endswith(b.EXTENSION):
                    return b.from_file(path, num_workers=num_workers)
            raise NotImplementedError(f'Extension of the dataset not available in {cls.DATASET}')
        else:
            return cls.DATASET.from_file(path, num_workers=num_workers)
        
    @property
    def TARGET_FIELDS(self) -> List[str]:
        return [tkz.name for tkz in self.target_tkzs]
    
    @classmethod
    def transform_data(cls, data: AbstractDataset, *args):
        for sen in tqdm(data, total=len(data), desc=f'transform-{data.name}', leave=False):
            cls.transform(sen, *args)
            
    def track(
        self, 
        data: Union[AbstractDataset, str],
        batch_size: int, 
        path: Optional[str] = None,
        epochs: int = 200,
        num_workers: int = 1,
        max_norm: float = 2.0
    ):
        """Tracks the carbon footprint in prediction and training.

        Args:
            data (Union[AbstractDataset, str]): Input dataset.
            batch_size (int): Batch size.
            path (Optional[str]): Path to store the logs.
            num_workers (int): Parallelization threads.
        """
        self.model.train()
        data = self.build_data(data, num_workers)
        loader = data.loader(self.collate, batch_size=batch_size, pred=False)
        
        tracker = CarbonTracker(epochs=epochs, log_dir=path, log_file_prefix='carbon', ignore_errors=True, verbose=1)
        tracker.epoch_start()
        with bar(total=len(data), desc='train-epoch', leave=True) as pbar:
            for inputs, targets, masks, sens in loader:
                _loss, _ = self.train_step(*to(self.device, inputs, targets, masks))
                
                # backward pass 
                self.optimizer.zero_grad()
                _loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
                self.optimizer.step()
                
                # update pbar 
                pbar.update(len(sens))
        tracker.epoch_end()
        tracker.stop()
        if path:
            self.clean_track(f'{path}/train-carbon.log')
        
        self.model.eval()
        tracker = CarbonTracker(epochs=1, log_dir=path, log_file_prefix='carbon', ignore_errors=True, verbose=1)
        tracker.epoch_start()
        with bar(total=len(data), desc='predict-epoch', leave=True) as pbar:
            for inputs, targets, masks, sens in loader:
                self.pred_step(*to(self.device, inputs, masks), sens)
                pbar.update(len(sens))
        tracker.epoch_end()
        tracker.stop()
        if path:
            self.clean_track(f'{path}/predict-carbon.log')
        
    def clean_track(self, path: str):
        *folder, _ = path.split('/')
        folder = '/'.join(folder)
        for file in os.listdir(folder):
            if file.startswith('carbon') and file.endswith('output.log'):
                os.rename(f'{folder}/{file}', path)
            if file.startswith('carbon'):
                os.remove(f'{folder}/{file}')