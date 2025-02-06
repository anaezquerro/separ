from torch.utils.data import Sampler 
import random, torch 
from typing import List, Tuple

class TokenizedSampler(Sampler):
    def __init__(self, data, batch_size: int, shuffle: bool = True, n_buckets: int = 10):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data 
        _, self.buckets = kmeans(data.lens, n_buckets)
        self.step()
        
    def __iter__(self):
        batch, total = [], 0
        for bucket in self.buckets:
            for i in bucket:
                length = len(self.data[i])
                if ((total + length) > self.batch_size and len(batch) > 0):
                    yield batch 
                    batch, total = [], 0
                batch.append(i)
                total += length 
        yield batch
        
    def __len__(self) -> int:
        return len(list(iter(self)))

    def step(self):
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(bucket)
            random.shuffle(self.buckets)
            
            
class StrictTokenizationSampler(Sampler):
    def __init__(self, data, batch_size: int, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data 
        self.step()
        buckets = {}
        for i, sen in enumerate(data):
            try:
                buckets[len(sen)].append(i)
            except KeyError:
                buckets[len(sen)] = [i]
        self.buckets = list(buckets.values())
        
    def __iter__(self):
        batch, total = [], 0
        for bucket in self.buckets:
            for i in bucket:
                length = len(self.data[i])
                if ((total + length) > self.batch_size and len(batch) > 0):
                    yield batch 
                    batch, total = [], 0
                batch.append(i)
                total += length 
        yield batch
        
    def step(self):
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(bucket)
            random.shuffle(self.buckets)
            
    def __len__(self) -> int:
        return len(list(iter(self)))
            
def kmeans(x: List[int], k: int, max_it: int = 32) -> Tuple[List[float], List[List[int]]]:
    r"""
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (List[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters, which is an approximate value.
            The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        List[float], List[List[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10, 20, (10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    x = torch.tensor(x, dtype=torch.float)
    # collect unique datapoints
    datapoints, indices, freqs = x.unique(return_inverse=True, return_counts=True)
    # the number of clusters must not be greater than the number of datapoints
    k = min(len(datapoints), k)
    # initialize k centroids randomly
    centroids = datapoints[torch.randperm(len(datapoints))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        for i in none:
            # the biggest cluster
            biggest = torch.where(mask[mask.sum(-1).argmax()])[0]
            # the datapoint farthest from the centroid of the biggest cluster
            farthest = dists[biggest].argmax()
            # update the assigned cluster of the farthest datapoint
            y[biggest[farthest]] = i
            # re-calculate the mask
            mask = torch.arange(k).unsqueeze(-1).eq(y)
        # update the centroids
        centroids, old = (datapoints * freqs * mask).sum(-1) / (freqs * mask).sum(-1), centroids
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)
        # stop iteration early if the centroids converge
        if centroids.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = centroids[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(indices.unsqueeze(-1).eq(torch.where(y.eq(i))[0]).any(-1))[0].tolist() for i in assigned]

    return centroids, clusters