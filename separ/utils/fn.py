from typing import Iterator, List, Callable, Union, Dict, Iterable, Optional
import torch, random, os, shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch.nn.utils.rnn import PackedSequence
from transformers import AutoConfig
from tqdm import tqdm 

def flatten(*lists, levels: int = -1) -> list:
    result = []
    for item in lists:
        if (isinstance(item, list) or isinstance(item, tuple) or isinstance(item, Iterator) or isinstance(item, set)) \
            and levels != 0:
            result += flatten(*item, levels=levels-1)
        else:
            result.append(item)
    return result 

def to(device: str, *tensors):
    result = []
    for x in tensors:
        if isinstance(x, torch.Tensor) or isinstance(x, PackedSequence):
            result.append(x.to(device, non_blocking=True))
        else:
            result.append(to(device, *x))
    return result 

def pad2D(tensors: Iterator[torch.Tensor], value: int = 0) -> torch.Tensor:
    tensors = list(tensors)
    max_len0, max_len1 = map(max, zip(*[tuple(x.shape) for x in tensors]))
    for i, x in enumerate(tensors):
        x = torch.cat([x, torch.full((x.shape[0], max_len1-x.shape[1]), value)], -1)
        x = torch.cat([x, torch.full((max_len0-x.shape[0], max_len1), value)], 0)
        tensors[i] = x
    return torch.stack(tensors, 0)


def create_2d_mask(sens, bos: bool = False, eos: bool = False) -> torch.Tensor:
    max_len = max(map(len, sens)) + bos + eos
    mask = torch.zeros(len(sens), max_len, max_len, dtype=torch.bool)
    for i, s in enumerate(sens):
        mask[i, :(len(s) + bos + eos), :(len(s) + bos + eos)] = True 
    return mask 

def onehot(values: List[int], num_classes: int) -> torch.Tensor:
    x = torch.zeros(num_classes, dtype=torch.bool)
    x[values] = True
    return x


def create_mask(lens: List[int], bos: bool = False, eos: bool = False) -> torch.Tensor:
    max_len = max(lens)
    mask = torch.zeros(len(lens), max_len, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :l] = True 
    if bos:
        mask = torch.concat([torch.ones(len(lens), 1, dtype=torch.bool), mask], 1)
    if eos:
        mask = torch.concat([torch.ones(len(lens), 1, dtype=torch.bool), mask], 1)
    return mask 


def split(items, lens: Union[List[int], int]):
    if isinstance(lens, int):
        lens = [lens for _ in range(len(items)//lens)]
    assert len(items) == sum(lens), f'Trying to split a list of {len(items)} in {sum(lens)}'
    result = []
    for l in lens:
        result.append([items.pop(0) for _ in range(l)])
    return result
    
    
    
def mmap(func: Callable, name: str, *data):
    data = list(zip(*data))
    return [func(*x) for x in tqdm(data, total=len(data), desc=name, leave=False)]
    
def parallel(func: Callable, *data, num_workers: int = os.cpu_count(), name: str = ''):
    if num_workers <= 1:
        return mmap(func, name, *data)
    results = []
    min_len = min(map(len, data))
    num_workers = min(num_workers, min_len)
    batch_size = int(min_len//num_workers+0.5)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for i in range(0, min_len, batch_size):
            partial = [x[i:(i+batch_size)] for x in data]
            futures.append(pool.submit(mmap, func, name, *partial))
        for f in futures:
            results += f.result()
    return results 


def init_folder(path: str):
    """Creates a new folder (removes the existing one).

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)
    
def acc(scores: torch.Tensor, gold: torch.Tensor) -> float:
    if len(scores.shape) > len(gold.shape):
        return (1.*(scores.argmax(-1) == gold)).mean().item()*100
    else:
        return (1.*(scores == gold)).mean().item()*100

def fscore(scores: torch.Tensor, gold: torch.Tensor, threshold: float = 0.0) -> float:
    gold = gold.to(torch.bool)
    pred = scores > threshold
    tp = (pred & gold).sum()
    fp = (pred & ~gold).sum()
    fn = (~pred & gold).sum()
    rec = div(tp, tp+fn)
    prec = div(tp, tp+fp)
    return div(2*rec*prec, rec+prec)*100


def listdir(folder: str, absolute: bool = False) -> List[str]:
    paths = os.listdir(folder)
    if absolute:
        paths = [f'{folder}/{path}' for path in paths]
    return paths 

def rfiles(folder: str) -> List[str]:
    paths = []
    for file in os.listdir(folder):
        if os.path.isdir(f'{folder}/{file}'):
            paths += rfiles(f'{folder}/{file}')
        else:
            paths.append(f'{folder}/{file}')
    return paths 

def remove(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    
def div(num: Union[torch.Tensor, float], dem: Union[torch.Tensor, float]) -> float:
    if dem == 0:
        return 0 
    else:
        return float(num/dem)
    
    
def isum(item1: Union[Iterable[float], float], item2: Union[Iterable[float], float]):
    if isinstance(item1, Iterable) and isinstance(item2, Iterable):
        return type(item1)(v1 + v2 for v1, v2 in zip(item1, item2))
    else:
        return item1 + item2
    
def merge(d1: dict, d2: dict, func: Callable = isum) -> dict:
    res = dict()
    for key in d1.keys() & d2.keys():
        res[key] = func(d1[key], d2[key]) 
    for key in d1.keys() - d2.keys():
        res[key] = d1[key]
    for key in d2.keys() - d1.keys():
        res[key] = d2[key]
    return res

def fdict(d: Dict[str, Union[float, Iterable[float]]], factor: float = 1, n: int = 3, prefix: str = '') -> str:
    """Formats a dictionary of numeric values.

    Args:
        d (Dict[str, Union[float, Iterable[float]]]): Input dictionary.
        scale (float, optional): Scale of values. Defaults to 1.
        n (int, optional): Number of decimals shown per value. Defaults to 3.
        prefix (str, optional): Prefix added to the formatted representation. Defaults to ''.

    Returns:
        str: Formatted dictionary.

    Examples:
    >>> a = dict(a=0.75, b=0.34, c=0.11)
    >>> fdict(a, scale=10, n=1)
    'a=7.5, b=3.4, c=1.1'
    """
    if len(d) == 0:
        return ''
    else:
        return prefix + ', '.join(f'{name}={apply(value, lambda x: round(x*factor, n))}' for name, value in d.items())

def avg(nums) -> int:
    """Average of a numerical sequence."""
    summ, count = 0, 0
    for x in nums:
        summ += x 
        count += 1
    return summ/count 

def scale_dict(vals: Dict[str, Union[float, Iterable[float]]], factor: float) -> Dict[str, Union[float, Iterable[float]]]:
    return {key: apply(value, lambda x: x*factor) for key, value in vals.items()}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    
def unbatch(x: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    return x[mask].split(mask.sum(-1).tolist())

def max_len(pretrained: str, default: int = 512) -> int:
    config = AutoConfig.from_pretrained(pretrained)
    if config.max_position_embeddings > 1:
        return config.max_position_embeddings
    else:
        return default 
    
def apply(seq, func: Callable):
    if isinstance(seq, Iterable):
        return type(seq)(func(x) for x in seq)
    else:
        return func(seq)
    
def filename(path: str) -> str:
    """Get the filename of a path.
    
    Examples:
        >>> filename('configs/lstm.ini')
        lstm.ini
    """
    *folder, file = path.split('/')
    return file 

