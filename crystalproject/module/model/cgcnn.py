import torch
import torch.nn as nn
import numpy as np

from crystalproject.utils.registry import registry
from crystalproject.module.model.intra_conv_layer import IntraConvLayer
from crystalproject.module.utils.node_embedding import NodeEmbedding
from crystalproject.module.utils.edge_embedding import EdgeEmbedding


# 预测器
@registry.register_model("cgcnn")
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        node_embedding={},
        edge_embedding={"dmin": 0.0, "dmax": 8.0, "step": 0.2},
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
    ):
        """
        Initialize CrystalGraphConvNet.

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
        super(CrystalGraphConvNet, self).__init__()
        self.ari = NodeEmbedding(**node_embedding)
        self.gd = EdgeEmbedding(**edge_embedding)
        orig_atom_fea_len = self.ari.get_dim()
        nbr_fea_len = self.gd.get_dim()
        self.embedding = nn.Linear(
            orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList(
            [
                IntraConvLayer(
                    node_fea_len=atom_fea_len,
                    edge_fea_len=nbr_fea_len
                )
                for _ in range(n_conv)
            ]
        )
        self.fc = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList(
                [nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 1)

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
        # 嵌入，卷积
        atom_fea = self.ari(atom_fea)
        nbr_fea = self.gd(nbr_fea)
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        # 池化
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # 输出
        crys_fea = self.softplus(self.fc(self.softplus(crys_fea)))
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out

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


if __name__ == "__main__":
    model_cls = registry.get_model_class("cgcnn")
    model = model_cls(64, 64)
    print()
