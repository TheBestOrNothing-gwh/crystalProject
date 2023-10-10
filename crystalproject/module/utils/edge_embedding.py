import torch
import torch.nn as nn
import numpy as np


class EdgeEmbedding(nn.Module):
    def __init__(self, dmin, dmax, step, var=None):
        super(EdgeEmbedding, self).__init__()
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = nn.Parameter(torch.tensor(
            np.arange(dmin, dmax, step), dtype=torch.float32), requires_grad=False)
        if var is None:
            var = step ** 2
        self.var = nn.Parameter(torch.tensor(
            var, dtype=torch.float32), requires_grad=False)
        self.dim = self.filter.size(0)

    def get_dim(self):
        return self.dim

    def forward(self, x):
        return torch.exp(
            -torch.pow(x - self.filter, 2) / self.var
        )
