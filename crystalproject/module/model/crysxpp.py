import torch
import torch.nn as nn

from crystalproject.utils.registry import registry
from crystalproject.module.model.intra_conv_layer import IntraConvLayer
from crystalproject.module.utils.node_embedding import NodeEmbedding
from crystalproject.module.utils.edge_embedding import EdgeEmbedding


# autoencoder
@registry.register_model("crysAE")
class CrystalAE(nn.Module):
    """
    Create a crystal graph auto encoder to learn node representations through unsupervised training
    """

    def __init__(
        self,
        node_embedding={},
        edge_embedding={"dmin": 0.0, "dmax": 8.0, "step": 0.2},
        atom_fea_len=64,
        n_conv=3,
    ):
        super(CrystalAE, self).__init__()
        self.ari = NodeEmbedding(**node_embedding)
        self.gd = EdgeEmbedding(**edge_embedding)
        orig_atom_fea_len = self.ari.get_dim()
        nbr_fea_len = self.gd.get_dim()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList(
            [
                IntraConvLayer(node_fea_len=atom_fea_len,
                               edge_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc_adj = nn.Bilinear(atom_fea_len, atom_fea_len, 6)
        self.fc1 = nn.Linear(6, 6)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc_atom_feature = nn.Linear(atom_fea_len, orig_atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        # Encoder part
        atom_fea = self.ari(atom_fea)
        nbr_fea = self.gd(nbr_fea)
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        # Decoder part
        """
        Architecture: Node emb is N*64. We decode it back to adjacency tensor N*N*4.
        Entry (i,j) is a 4 dim one hot vector,
        where 0th vector = 1 -> No edges
        1st vector = 1 -> 1/2 Edges
        2nd vector = 1 -> 3/4 Edges
        3rd vector = 1 -> more than 5 Edges
        """
        adj_list = []
        for idx_map in crystal_atom_idx:
            N, dim = atom_fea[idx_map].shape
            adj = atom_fea[idx_map].repeat(N, 1, 1).contiguous().view(-1, dim)
            adj_list.append(adj)
        adj = torch.cat(adj_list, dim=0)
        adj = self.fc_adj(adj, adj)
        adj = self.fc1(adj)
        adj = self.logsoftmax(adj)
        atom_fea = self.fc_atom_feature(atom_fea)
        return adj, atom_fea
