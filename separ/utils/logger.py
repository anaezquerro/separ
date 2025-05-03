import logging.handlers
import logging
from tqdm import tqdm  
from typing import Optional, Iterable

class CustomFormatter(logging.Formatter):
    blue = "\x1b[1;36m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    purple = "\x1b[35;20m"
    magenta = "\x1b[1;35m"

    FORMATS = {
        logging.DEBUG: blue +  '%(message)s' + reset,
        logging.INFO: green + '[%(asctime)s] ' + reset + '%(message)s',
        logging.WARNING: yellow + '%(message)s' + reset,
        logging.ERROR: red + '[%(asctime)s] ' + reset + '%(message)s',
        logging.CRITICAL: bold_red + '[%(asctime)s] ' + reset + '%(message)s'
    }
    
    DIST_FORMATS = {
        logging.DEBUG: blue +  '%(message)s' + reset,
        logging.INFO: magenta + '[%(asctime)s] ' + reset + '%(message)s',
        logging.WARNING: yellow + '%(message)s' + reset,
        logging.ERROR: red + '[%(asctime)s] ' + reset + '%(message)s',
        logging.CRITICAL: bold_red + '[%(asctime)s] ' + reset + '%(message)s'
    }
    
    def __init__(self, dist: bool = False):
        self.FORMAT = self.FORMATS if not dist else self.DIST_FORMATS

    def format(self, record):
        log_fmt = self.FORMAT.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt= "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def logger(
    name: str, 
    path: Optional[str] = None, 
    level = logging.DEBUG,
    dist: bool = False
) -> logging.Logger:
    log = logging.getLogger(name)
    if log.hasHandlers():
        log.handlers.clear()
    log.propagate = False
    log.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter(dist=dist))
    log.addHandler(ch)
    if path is not None:
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    return log 


def bar(iterable: Optional[Iterable] = None, **kwargs):
    return tqdm(iterable, bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]', **kwargs)
    
def bar_postfix(*args: str, **kwargs) -> str:
    return ', '.join(list(args) + [f'{key}={value}' for key, value in kwargs.items()])

    