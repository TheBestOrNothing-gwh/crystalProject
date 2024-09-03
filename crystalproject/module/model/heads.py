import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot_orthogonal

from crystalproject.utils.registry import registry


@registry.register_head("mlphead")
class MLPhead(nn.Module):

    def __init__(self, in_channels=256, hidden_channels=128, out_channels=1, targets=[], descriptors=[]):
        super(MLPhead, self).__init__()
        assert len(targets) == out_channels, "输出维度必须等于目标数量"
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Sigmoid(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.targets = targets
        self.descriptors = descriptors
    
    def reset_parameters(self):
        glorot_orthogonal(self.mlp.weight, scale=2.0)

    def forward(self, batch_data):
        input = torch.cat([batch_data[descriptor] for descriptor in self.descriptors], dim=1)
        out = self.mlp(input)
        for i, target in enumerate(self.targets):
            batch_data["output"][target] = out[:, [i]]

@registry.register_head("lin")
class Linearhead(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, targets=[], descriptors=[]):
        super(Linearhead, self).__init__()
        assert len(targets) == out_channels, "输出维度必须等于目标数量"
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.targets = targets
        self.descriptors = descriptors
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, batch_data):
        input = torch.cat([batch_data[descriptor] for descriptor in self.descriptors], dim=1)
        out = self.lin(input)
        for i, target in enumerate(self.targets):
            batch_data["output"][target] = out[:, [i]]
        