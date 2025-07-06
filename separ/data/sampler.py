from torch.utils.data import Sampler
import random, os

from separ.utils.fn import interleave
from separ.utils.shard import is_distributed, local_rank
from separ.utils.common import WORLD_SIZE


class TokenizedBatchSampler(Sampler):
    def __init__(
        self, 
        lens: dict[int, int], 
        batch_size: int, 
        shuffle: bool = False
    ):
        """
        Token-based batchification.
        
        Args:
            lens (Dict[int, int]): Dictionary of the length of each input (index -> length).
            batch_size (int): Number of tokens that are passed at each batch.
            shuffle (bool): Whether to shuffle the dataset.
            device (Optional[int]): Configure if only a partial view of the dataset is passed (distributed training). 
            
        """
        self.lens = lens
        self.lmap = {length: [] for length in set(self.lens.values())}
        for index, length in lens.items():
            self.lmap[length].append(index)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.set_epoch(0)
        
    def generate_indices(self):
        """
        Generates the indices variable. In shuffling mode, it shuffles the indices corresponding
        to samples with the same length.
        """
        indices = []
        for length in sorted(self.lmap.keys()):
            if self.shuffle:
                random.shuffle(self.lmap[length])
            indices += self.lmap[length]
        self._indices = indices 
        if is_distributed():
            self._indices = self._indices[local_rank():len(self.lens):WORLD_SIZE]
        
    def generate_batches(self):
        """
        Generates the batches variable. In shuffling mode, it randomly selects an element from 
        the indices array and then selects contiguous values until the batch size is reached.
        """
        indices = self.indices
        batches = []
        if self.shuffle:
            if is_distributed(): # ensure that similar lengths in each device are selected
                random.seed(self.epoch)
            while len(indices) > 0:
                center = random.randint(0, len(indices)-1)
                batch = [indices[center]]
                for nextt in interleave(indices[:center][::-1], indices[center+1:]):
                    if sum(map(self.lens.get, batch)) + self.lens[nextt] > self.batch_size:
                        break 
                    else:
                        batch.append(nextt)
                batches.append(batch)
                indices = [i for i in indices if i not in batch]
        else:
            batch = []
            while len(indices) > 0:
                if (sum(map(self.lens.get, batch)) + self.lens[indices[0]]) > self.batch_size:
                    if len(batch) == 0:
                        batch.append(indices.pop(0))
                    batches.append(batch)
                    batch = []
                else:
                    batch.append(indices.pop(0))
            if len(batch) > 0:
                batches.append(batch)
        self._batches = batches 
    
    @property
    def indices(self) -> list[int]:
        """
        Generates the list of dataset indices maintaining length order.
        
        Examples:
            >>> lens = {1:10, 2:5, 3:15, 4:3, 5:10}
            >>> sampler = TokenizedBatchSampler(lens, batch_size=10, shuffle=True)
            >>> sampler.indices
            [4, 2, 5, 1, 3]
        """
        return self._indices.copy()

    @property
    def batches(self) -> list[list[int]]:
        """Generate the batches.

        Returns:
            list[list[int]]: List of batches.
        """
        return self._batches.copy()
                
    def set_epoch(self, epoch: int):
        self.epoch = epoch 
        if self.shuffle or epoch == 0:
            self.generate_indices()
            self.generate_batches()
        
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self) -> int:
        return len(self.batches)
    
    @property
    def num_sens(self) -> int:
        return sum(map(len, self.batches))
    

    @property 
    def num_tokens(self) -> int:
        return sum(map(self.lens.get, self.indices))