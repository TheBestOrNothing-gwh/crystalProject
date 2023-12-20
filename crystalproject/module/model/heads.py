import torch
import torch.nn as nn
from crystalproject.utils.registry import registry


@registry.register_head("regression")
class Reghead(nn.Module):

    def __init__(self, in_channels=256, hidden_channels=128, out_channels=1, targets=[], descriptors=[]):
        super(Reghead, self).__init__()
        assert len(targets) == out_channels, "输出维度必须等于目标数量"
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Sigmoid(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.targets = targets
        self.descriptors = descriptors

    def forward(self, batch_data):
        input = torch.cat([batch_data[descriptor] for descriptor in self.descriptors], dim=1)
        out = self.mlp(input)
        for i, target in enumerate(self.targets):
            batch_data["output"][target] = out[:, [i]]
        
        