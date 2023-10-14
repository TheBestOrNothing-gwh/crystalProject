import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.conv import GATv2Conv

from crystalproject.utils.registry import registry
from crystalproject.module.utils.node_embedding import NodeEmbedding
from crystalproject.module.utils.edge_embedding import EdgeEmbedding


# 基于GAT_v2的预测器
@registry.register_model("gat_v2")
class CrystalGraphAttNet(nn.Module):
    """
    Create a crystal graph attention neural network
    for predicting total material properties.
    """

    def __init__(
        self,
        node_embedding={},
        edge_embedding={"dmin": 0.0, "dmax": 8.0, "step": 0.2},
        atom_fea_len=64,
        n_att=3
    ):
        """
        Initialize CrystalGraphAttNet.

        Parameters
        ----------
        orig_atom_fea_len: int
            Number of atom_features in the input.
        nbr_fea_len: int
            Number of bond features
        atom_fea_len: int
            Number of hidden atom features in the convolutional layers
        n_conv: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_h: int
            Number of hidden layers after pooling
        """
        super(CrystalGraphAttNet, self).__init__()
        self.ari = NodeEmbedding(**node_embedding)
        self.gd = EdgeEmbedding(**edge_embedding)
        orig_atom_fea_len = self.ari.get_dim()
        nbr_fea_len = self.gd.get_dim()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.atts = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=atom_fea_len,
                    out_channels=atom_fea_len,
                    edge_dim=nbr_fea_len
                )
                for _ in range(n_att)
            ]
        )

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
            Atom features from atom type
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        """
        # 嵌入，注意力
        atom_fea = self.ari(atom_fea)
        nbr_fea = self.gd(nbr_fea)
        atom_fea = self.embedding(atom_fea)
        for att in self.atts:
            atom_fea = att(atom_fea, nbr_fea_idx, nbr_fea)
        # 池化
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        return crys_fea

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom feature vectors of the batch
        crystal_atom_idx: List of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx
        """
        assert (
            np.sum(crystal_atom_idx)
            == atom_fea.data.shape[0]
        )
        crystal_feas = torch.split(atom_fea, crystal_atom_idx.tolist(), 0)
        crystal_feas = [torch.mean(crystal_fea, dim=0, keepdim=True)
                        for crystal_fea in crystal_feas]
        return torch.cat(crystal_feas, dim=0)
