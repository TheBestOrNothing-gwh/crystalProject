import torch
import torch.nn as nn
from torch_geometric.nn import SimpleConv
from torch_scatter import scatter

from crystalproject.utils.registry import registry


@registry.register_model("gcn")
class GCN(nn.Module):
    """
    GCN
    """
    def __init__(
        self,
        num_layers=2,
        hidden_channels=128,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList([SimpleConv(combine_root="self_loop") for _ in range(num_layers)])

    def forward(self, batch_data):
        v, edge_index = batch_data["v"], batch_data["edges"]
        for conv in self.convs:
            v = conv(v, edge_index)
        return v
