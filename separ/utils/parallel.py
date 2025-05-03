from typing import Callable 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 
import os 
    
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