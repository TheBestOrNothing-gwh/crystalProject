import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from crystalproject.utils.registry import registry
from crystalproject.module.utils.dist_embedding import DistEmbedding


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


# 预测器
@registry.register_model("cgcnn")
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        edge_embedding={"dmin": 0.0, "dmax": 8.0, "step": 0.2},
        hidden_channels=64,
        num_layers=3
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------
        orig_hidden_channels: int
            Number of atom_features in the input.
        nbr_fea_len: int
            Number of bond features
        hidden_channels: int
            Number of hidden atom features in the convolutional layers
        num_layers: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_h: int
            Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.gd = DistEmbedding(**edge_embedding)
        nbr_fea_len = self.gd.get_dim()
        self.convs = nn.ModuleList(
            [
                IntraConvLayer(
                    node_fea_len=hidden_channels,
                    edge_fea_len=nbr_fea_len
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch_data):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_hidden_channels)
            Atom features from atom type
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            indices of M neighbors of each atom
        batch_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        """
        # 嵌入，卷积
        atom_fea, pos, nbr_fea_idx, offsets_real = batch_data["v"], batch_data["pos"], batch_data["edges"], batch_data["offsets_real"]
        
        row, col = nbr_fea_idx
        dist = ((pos[col] + offsets_real) - pos[row]).norm(dim=-1)
        nbr_fea = self.gd(dist)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        return atom_fea