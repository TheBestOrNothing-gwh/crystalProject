import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, mean=0, std=1):
        super(Normalizer, self).__init__()
        self._norm_func = lambda tensor: (tensor - mean) / std
        self._denorm_func = lambda tensor: tensor * std + mean

        self.mean = nn.Parameter(torch.tensor(
            mean, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(
            std, dtype=torch.float32), requires_grad=False)

    def norm(self, tensor):
        return self._norm_func(tensor)

    def denorm(self, tensor):
        return self._denorm_func(tensor)
