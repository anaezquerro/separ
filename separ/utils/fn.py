from typing import Iterator, List, Union
from transformers import AutoConfig
import torch, os, tempfile, shutil, random, pickle 
import numpy as np 
from torch.nn.utils.rnn import PackedSequence

def flatten(*lists, levels: int = -1) -> list:
    result = []
    for item in lists:
        if (isinstance(item, list) or isinstance(item, tuple) or isinstance(item, Iterator) or isinstance(item, set)) \
            and levels != 0:
            result += flatten(*item, levels=levels-1)
        else:
            result.append(item)
    return result 
    
def to(device: int, *tensors) -> List[torch.Tensor]:
    out = []
    for x in tensors:
        if isinstance(x, torch.Tensor) or isinstance(x, PackedSequence):
            out.append(x.to(device))
        else:
            out.append(to(device, *x))
    return out 



def interleave(l1: list, l2: list) -> list:
    """Interleave elements of two lists."""
    res = []
    for x1, x2 in zip(l1, l2):
        res += [x1, x2]
    if len(l1) > len(l2):
        res += l1[len(l2):]
    elif len(l2) > len(l1):
        res += l2[len(l1):]
    return res 


def invert_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def merge_dicts(d1: dict, d2: dict) -> dict:
    """Merges two dictionaries adding numeric values."""
    res = d1 | d2
    for common in d1.keys() & d2.keys(): # merge common keys 
        res[common] = d1[common] + d2[common]
    return res 
    
    
def mkdtemp() -> str:
    tmp_folder = tempfile.mkdtemp()
    shutil.rmtree(tmp_folder)
    tmp_folder = '.tmp/' + tmp_folder.split('/')[-1]
    os.makedirs(tmp_folder)
    return tmp_folder

def mkftemp() -> str:
    tmp_folder = tempfile.mkdtemp()
    shutil.rmtree(tmp_folder)
    tmp_path = '.tmp/' + tmp_folder.split('/')[-1]
    return tmp_path

def div(num: Union[torch.Tensor, float], dem: Union[torch.Tensor, float]) -> float:
    if dem == 0 and num > 0:
        return 0 
    elif dem == 0 and num == 0:
        return 1.0
    else:
        return float(num/dem)
    
def folderpath(path: str) -> str:
    return '/'.join(path.split('/')[:-1])


def acc(preds: torch.Tensor, golds: torch.Tensor) -> torch.Tensor:
    """Computes the accuracy between scores/predictions and targets.

    Args:
        preds (torch.Tensor): Scores or predictions.
        golds (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Accuracy.
    """
    if preds.ndim > golds.ndim:
        preds = preds.argmax(-1)
    return (100.*(preds.to(torch.int32) == golds.to(torch.int32))).mean().detach()


def split(items, lens: Union[List[int], int]):
    if isinstance(lens, int):
        lens = [lens for _ in range(len(items)//lens)]
    assert len(items) == sum(lens), f'Trying to split a list of {len(items)} in {sum(lens)}'
    result = []
    for l in lens:
        result.append([items.pop(0) for _ in range(l)])
    return result
    
def max_len(pretrained: str, default: int = 512) -> int:
    config = AutoConfig.from_pretrained(pretrained)
    try:
        if config.max_position_embeddings > 1:
            return config.max_position_embeddings
        else:
            return default 
    except AttributeError:
        return default
    

def avg(nums) -> int:
    """Average of a numerical sequence."""
    summ, count = 0, 0
    for x in nums:
        summ += x 
        count += 1
    return summ/count 


def get_mask(lens: List[int], bos: bool = False, eos: bool = False) -> torch.Tensor:
    """Gets a padding mask from the list of input lengths."""
    mask = torch.zeros(len(lens), max(lens) + bos + eos, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :(l+bos+eos)] = True 
    return mask 

def filename(path: str, extension: bool = True) -> str:
    name = path.split('/')[-1]
    if not extension:
        name = '.'.join(name.split('.')[:-1])
    return name

def change_extension(path: str, new_ext: str) -> str:
    return '.'.join(path.split('.')[:-1] + [new_ext])

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def listdir(path: str, absolute: bool = False) -> List[str]:
    if absolute:
        return [f'{path}/{file}' for file in os.listdir(path)]
    else:
        return os.listdir(path)
    
def remove(path: str):
    if not os.path.exists(path):
        return
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    
    
def pad2D(tensors: Iterator[torch.Tensor], value: int = 0) -> torch.Tensor:
    tensors = list(tensors)
    max_len0, max_len1 = map(max, zip(*[tuple(x.shape) for x in tensors]))
    for i, x in enumerate(tensors):
        x = torch.cat([x, torch.full((x.shape[0], max_len1-x.shape[1]), value, device=x.device)], -1)
        x = torch.cat([x, torch.full((max_len0-x.shape[0], max_len1), value, device=x.device)], 0)
        tensors[i] = x
    return torch.stack(tensors, 0)


def get_2d_mask(lens: List[int], bos: bool = False, eos: bool = False) -> torch.Tensor:
    max_len = max(lens) + bos + eos
    mask = torch.zeros(len(lens), max_len, max_len, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :(l + bos + eos), :(l + bos + eos)] = True 
    return mask 

def fscore(scores: torch.Tensor, gold: torch.Tensor, threshold: float = 0.0) -> float:
    gold = gold.to(torch.bool)
    pred = scores > threshold
    tp = (pred & gold).sum()
    fp = (pred & ~gold).sum()
    fn = (~pred & gold).sum()
    rec = div(tp, tp+fn)
    prec = div(tp, tp+fp)
    return div(2*rec*prec, rec+prec)*100

def macro_fscore(scores: torch.Tensor, gold: torch.Tensor, threshold: float = 0.0, exclude: List[int] = [])  -> float:
    if len(scores.shape) > len(gold.shape):
        return avg(fscore(score, gold == i, threshold) for i, score in enumerate(scores.unbind(-1)) if (gold == i).any() and i not in exclude)
    elif len(scores.shape) == len(gold.shape):
        return avg(fscore(scores == i, gold == i, threshold=0) for i in gold.unique().tolist() if i not in exclude)
    else:
        raise NotImplementedError

def save(obj, path: str):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer, protocol=pickle.HIGHEST_PROTOCOL)