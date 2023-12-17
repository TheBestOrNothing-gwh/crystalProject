import torch.nn as nn
from crystalproject.utils.registry import registry


@registry.register_head("regression")
class Reghead(nn.Module):

    def __init__(self, in_channels=256, descriptor="underling_network_embedding", out_channels=1, targets=[]):
        super(Reghead, self).__init__()
        assert len(targets) == out_channels, "输出维度必须等于目标数量"
        self.fc = nn.Linear(in_channels, out_channels)
        self.descriptor = descriptor
        self.targets = targets

    def forward(self, batch_data):
        out = self.fc(batch_data[self.descriptor])
        for i, target in enumerate(self.targets):
            batch_data["output"][target] = out[:, i]
        
        