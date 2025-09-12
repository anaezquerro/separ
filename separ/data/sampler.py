# from torch.utils.data import Sampler
# import random, torch 

# from separ.utils.shard import is_distributed, local_rank, WORLD_SIZE, is_main, sync


# class TokenizedBatchSampler(Sampler):
#     def __init__(
#         self, 
#         lens: dict[int, int], 
#         batch_size: int, 
#         shuffle: bool = False,
#     ):
#         """
#         Token-based batchification.
        
#         Args:
#             lens (Dict[int, int]): Dictionary of the length of each input (index -> length).
#             batch_size (int): Number of tokens that are passed at each batch.
#             shuffle (bool): Whether to shuffle the dataset.
#             device (Optional[int]): Configure if only a partial view of the dataset is passed (distributed training). 
            
#         """
#         self.lens = lens
#         self.lmap = {length: [] for length in set(self.lens.values())}
#         for index, length in lens.items():
#             self.lmap[length].append(index)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.generate_indices()
#         self.set_epoch(0)
        
#     def generate_indices(self):
#         """
#         Generates the indices variable. In shuffling mode, it shuffles the indices corresponding
#         to samples with the same length.
#         """
#         if is_main():
#             indices = []
#             for length in sorted(self.lmap.keys()):
#                 if self.shuffle:
#                     random.shuffle(self.lmap[length])
#                 indices += self.lmap[length]
#         else:
#             indices = list(range(len(self.lmap)))
#         if is_distributed():
#             indices = torch.tensor(indices, device=local_rank())
#             sync(indices, rank=0)
#             indices = indices[local_rank()::WORLD_SIZE].tolist()
#         self._indices = indices
        
#     def generate_batches(self):
#         """
#         Generates the batches variable. In shuffling mode, it randomly selects an element from 
#         the indices array and then selects contiguous values until the batch size is reached.
#         """
#         indices = self.indices
#         batches = []
#         if self.shuffle:
#             while len(indices) > 0:
#                 center = random.randint(0, len(indices)-1)
#                 batch = [indices[center]]
#                 for nextt in interleave(indices[:center][::-1], indices[center+1:]):
#                     if sum(map(self.lens.get, batch)) + self.lens[nextt] > self.batch_size:
#                         break 
#                     else:
#                         batch.append(nextt)
#                 batches.append(batch)
#                 indices = [i for i in indices if i not in batch]
#         else:
#             batch = []
#             while len(indices) > 0:
#                 if (sum(map(self.lens.get, batch)) + self.lens[indices[0]]) > self.batch_size:
#                     if len(batch) == 0:
#                         batch.append(indices.pop(0))
#                     batches.append(batch)
#                     batch = []
#                 else:
#                     batch.append(indices.pop(0))
#             if len(batch) > 0:
#                 batches.append(batch)
#         self._batches = batches 
    
#     @property
#     def indices(self) -> list[int]:
#         """
#         Generates the list of dataset indices maintaining length order.
        
#         Examples:
#             >>> lens = {1:10, 2:5, 3:15, 4:3, 5:10}
#             >>> sampler = TokenizedBatchSampler(lens, batch_size=10, shuffle=True)
#             >>> sampler.indices
#             [4, 2, 5, 1, 3]
#         """
#         return self._indices.copy()

#     @property
#     def batches(self) -> list[list[int]]:
#         """Generate the batches.

#         Returns:
#             list[list[int]]: List of batches.
#         """
#         return self._batches.copy()
                
#     def set_epoch(self, epoch: int):
#         self.epoch = epoch 
#         if self.shuffle or epoch == 0:
#             self.generate_indices()
#             self.generate_batches()
        
#     def __iter__(self):
#         return iter(self.batches)
    
#     def __len__(self) -> int:
#         return len(self.batches)
    
#     @property
#     def num_sens(self) -> int:
#         return sum(map(len, self.batches))
    
#     @property 
#     def num_tokens(self) -> int:
#         return sum(map(self.lens.get, self.indices))

from torch.utils.data import Sampler
from separ.utils.shard import is_distributed, WORLD_SIZE, local_rank, is_main, sync
import random, torch 
from sklearn.cluster import KMeans 
import numpy as np 
import torch.distributed as dist 


def kmeans(x: np.ndarray, num_clusters: int) -> list[np.ndarray]:
    km = KMeans(num_clusters).fit(x.reshape(-1, 1))
    return [(km.labels_ == c).flatten() for c in range(num_clusters)]


class TokenizedBatchSampler(Sampler):
    def __init__(
        self,
        lens: dict[int,int],
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        num_buckets: int = 16
    ):
        super().__init__()
        self.lens = lens 
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.num_buckets = num_buckets
        self._seed = seed
        self.epoch = 0
         
        if is_main():
            # list of indices per length 
            self.lmap = {length: [] for length in set(lens.values())}
            for index, length in lens.items():
                self.lmap[length].append(index)

        self.indices = self.generate_indices()
        if not self.shuffle:
            self.batches = self.static_batches()
            self.sync()
        
    def set_epoch(self, epoch: int):
        self.epoch = epoch 
        if self.shuffle:
            random.seed(self.seed)
            self.indices = self.generate_indices()
            self.buckets = self.generate_buckets()
            self.batches = self.generate_batches()
            self.sync()
        
    def get_length(self, index):
        if isinstance(index, np.ndarray):
            return np.vectorize(self.lens.get)(index).flatten()
        elif isinstance(index, list):
            return list(map(self.lens.get, index))
        elif isinstance(index, torch.Tensor):
            return index.cpu().apply_(self.lens.get).to(index.device)
        else:
            return self.lens.get(index)
        
    @property
    def seed(self) -> int:
        return self.epoch + self._seed
        
    def generate_indices(self) -> np.ndarray:
        if is_main():
            indices = []
            for length in sorted(self.lmap.keys()):
                if self.shuffle:
                    random.shuffle(self.lmap[length])
                indices += self.lmap[length]
        else:
            indices = list(range(len(self.lens))) # initialize in distributed mode for other devices
        if is_distributed():
            indices = torch.tensor(indices, device=local_rank())
            sync(indices, rank=0)
            # n = (self.get_length(indices) <= self.batch_size).sum().item()
            indices = indices.tolist()
            if is_main():
                indices = indices[local_rank()::WORLD_SIZE]
                # + indices[n:]
            else:
                indices = indices[local_rank()::WORLD_SIZE]
        return np.array(indices)
    
    def generate_buckets(self) -> list[list[int]]:
        buckets = kmeans(self.get_length(self.indices), self.num_buckets)
        buckets = [self.indices[bucket].tolist() for bucket in buckets]
        random.shuffle(buckets)
        return buckets 
            
    def generate_batches(self) -> list[list[int]]:
        buckets = self.generate_buckets()
        batches = []
        while len(buckets) > 0:
            bucket = random.choice(buckets)
            batch = [bucket.pop(0)]
            while len(bucket) > 0 and sum(self.get_length(batch + [bucket[0]])) < self.batch_size:
                batch.append(bucket.pop(0))
            batches.append(batch)
            buckets = [b for b in buckets if len(b) > 0]
        return batches 
    
    def static_batches(self) -> list[list[int]]:
        batches = []
        batch = [self.indices[0].item()]
        for i in self.indices[1:].tolist():
            if sum(self.get_length(batch + [i])) > self.batch_size:
                batches.append(batch)
                batch = []
            batch.append(i)
        if len(batch) > 0:
            batches.append(batch)
        return batches 
    
    def sync(self):
        if is_distributed():
            n = torch.tensor(len(self.batches), device=local_rank())
            dist.all_reduce(n, op=dist.ReduceOp.MAX)
            batches = self.batches.copy()
            for i in range(len(self.batches), n.item()):
                batches.append(self.batches[i % len(self.batches)])
            self.batches = batches
        
    def __len__(self) -> int:
        return len(self.batches)
    
    def __iter__(self):
        return iter(self.batches)
    
    @property
    def num_tokens(self) -> int:
        return self.get_length(self.indices).sum().item()

    @property    
    def num_sens(self) -> int:
        return len(self.indices)
        