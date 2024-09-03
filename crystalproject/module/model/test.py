import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from crystalproject.utils.registry import registry
from crystalproject.module.utils.dist_embedding import DistEmbedding


# 预测器
@registry.register_model("test")
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self
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
        self.a = nn.Linear(16, 16)

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
        pos, nbr_fea_idx, offsets_real = batch_data["atom_bond_graph"]["pos"], batch_data["atom_bond_graph"]["edges"], batch_data["atom_bond_graph"]["offsets_real"]
        
        row, col = nbr_fea_idx
        dist = ((pos[col] + offsets_real) - pos[row]).norm(dim=-1)
        print(torch.max(dist))
        print(torch.min(dist))
        print("--------------")

        pos, nbr_fea_idx, offsets_real = batch_data["cluster_graph"]["pos"], batch_data["cluster_graph"]["edges"], batch_data["cluster_graph"]["offsets_real"]
        
        row, col = nbr_fea_idx
        dist = ((pos[col] + offsets_real) - pos[row]).norm(dim=-1)
        print(torch.max(dist))
        print(torch.min(dist))
        print(torch.sum(dist < 3))

        pos, nbr_fea_idx, offsets_real = batch_data["underling_network"]["pos"], batch_data["underling_network"]["edges"], batch_data["underling_network"]["offsets_real"]
        
        row, col = nbr_fea_idx
        dist = ((pos[col] + offsets_real) - pos[row]).norm(dim=-1)
        print(torch.max(dist))
        print(torch.min(dist))
        print(torch.sum(dist < 3))
        print("==============")
        