import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from crystalproject.utils.registry import registry


# simple conv Layer
@registry.register_model("cgcnn_layer")
class IntraConvLayer(MessagePassing):
    """
    Convolutional operation on graphs
    """

    def __init__(self, node_fea_len, edge_fea_len, aggr="add"):
        """
        Initialize ConvLayer

        Parameters
        ----------
        node_fea_len: int
            Number of node hidden features.
        edge_fea_len: int
            Number of edge features.
        aggr: str
            aggregate function.["add", "mean", "max"]
        """
        super(IntraConvLayer, self).__init__(aggr)
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
        self.fc_full = nn.Linear(
            2*self.node_fea_len+self.edge_fea_len, 2*self.node_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.node_fea_len)
        self.bn2 = nn.BatchNorm1d(self.node_fea_len)

    def forward(self, node_fea, edge_fea, edge_index):
        return self.propagate(edge_index, node_fea=node_fea, edge_fea=edge_fea)

    def message(self, node_fea_i, node_fea_j, edge_fea):
        total_fea = torch.cat([node_fea_i, node_fea_j, edge_fea], dim=1)
        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)
        nbr_filter, nbr_core = total_fea.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        return nbr_filter * nbr_core

    def update(self, aggr_out, node_fea):
        aggr_out = self.bn2(aggr_out)
        out = self.softplus(node_fea+aggr_out)
        return out
