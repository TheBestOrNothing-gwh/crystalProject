import torch
import torch.nn as nn

from crystalproject.utils.registry import registry

@registry.register_model("gaussian")
class Normalizer(nn.Module):
    def __init__(self, mean=0, std=1):
        super(Normalizer, self).__init__()
        self._norm_func = lambda tensor: (tensor - mean) / std
        self._denorm_func = lambda tensor: tensor * std + mean

    def norm(self, tensor):
        return self._norm_func(tensor)

    def denorm(self, tensor):
        return self._denorm_func(tensor)

@registry.register_model("max_min")
class Normalizer_MaxMin(nn.Module):
    def __init__(self, min=0, max=1):
        super(Normalizer_MaxMin, self).__init__()
        self._norm_func = lambda tensor: (tensor - min) / (max - min)
        self._denorm_func = lambda tensor: tensor * (max - min) + min

    def norm(self, tensor):
        return self._norm_func(tensor)

    def denorm(self, tensor):
        return self._denorm_func(tensor)
