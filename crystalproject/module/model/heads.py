import torch.nn as nn
from crystalproject.utils.registry import registry


@registry.register_head("mlphead")
class MLPhead(nn.Module):

    def __init__(self, fea_in_len=128, fea_len=128, n_h=1, classification=False):
        super(MLPhead, self).__init__()
        self.classification = classification
        self.fc0 = nn.Linear(fea_in_len, fea_len)
        self.softplus0 = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(fea_len, fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList(
                [nn.Softplus() for _ in range(n_h - 1)]
            )
        if self.classification:
            self.fc_out = nn.Linear(fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(fea_len, 1)
    
    def forward(self, fea):
        fea = self.softplus0(self.fc0(self.softplus0(fea)))
        if self.classification:
            fea = self.dropout(fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                fea = softplus(fc(fea))
        out = self.fc_out(fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out