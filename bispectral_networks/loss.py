import torch
import numpy as np


class LpDistance:
    def __init__(self, p=2, pairwise=False):
        self.p = p
        self.pairwise = pairwise

    def __call__(self, x1, x2):
        if x1.dtype == torch.complex64 or x1.dtype == torch.complex128:
            dtype = x1.real.dtype
        else:
            dtype = x1.dtype
        if self.pairwise:
            return torch.nn.functional.pairwise_distance(x1, x2, p=self.p)
        else:    
            rows, cols = np.meshgrid(range(len(x1)), range(len(x2)))
            rows = rows.flatten()
            cols = cols.flatten()
            dmat = torch.zeros((len(x1), len(x2)), dtype=dtype, device=x1.device)
            distances = torch.nn.functional.pairwise_distance(x1[rows], x2[cols], p=self.p)
            dmat[rows, cols] = distances
            return dmat
    
    
class OrbitCollapse(torch.nn.Module):
    
    def __init__(self,
                 distance=None):
        super().__init__()
        self.distance = distance
        
    def forward(self, embeddings, labels):
        L = 0
        count = 0
        for i in labels.unique():
            idx = torch.where(labels==i)[0]
            dmat = self.distance(embeddings[idx], embeddings[idx])
            ut_idx = np.triu_indices(dmat.shape[0], k=1)
            distances = dmat[ut_idx]
            L += distances.sum()
            count += len(distances)
        L /= count
        return L