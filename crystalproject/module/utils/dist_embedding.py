import torch
import torch.nn as nn
import numpy as np


class DistEmbedding(nn.Module):
    def __init__(self, dmin=0.0, dmax=8.0, step=0.2, var=None):
        super(DistEmbedding, self).__init__()
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
    

class schnetEmbedding(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(schnetEmbedding, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
