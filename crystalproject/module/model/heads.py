import torch.nn as nn
from crystalproject.utils.registry import registry


@registry.register_head("mlphead")
class MLPhead(nn.Module):

    def __init__(self, in_channels=256, hidden_channels=256, out_channels=1, n_h=1):
        super(MLPhead, self).__init__()
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.softplus0 = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(hidden_channels, hidden_channels) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList(
                [nn.Softplus() for _ in range(n_h - 1)]
            )
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, fea):
        fea = self.softplus0(self.fc0(self.softplus0(fea)))
        if self.classification:
            fea = self.dropout(fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                fea = softplus(fc(fea))
        out = self.fc_out(fea)
        return out