import torch.nn as nn
from torch_scatter import scatter

from crystalproject.utils.registry import registry
from crystalproject.module.utils.atom_embedding import AtomEmbedding


# 普通的晶体属性预测
@registry.register_model("radiusgraphnet")
class RadiusGraphNet(nn.Module):
    """
    """

    def __init__(
        self,
        atom_embedding={},
        atom_hidden_channels=128,
        atom_radius_graph={},
        reduce="mean",
    ):
        super(RadiusGraphNet, self).__init__()
        self.reduce = reduce
        self.atom_emb = AtomEmbedding(hidden_channels=atom_hidden_channels, **atom_embedding)
        model_cls = registry.get_model_class(atom_radius_graph["name"])
        self.atom_radius_graph = model_cls(hidden_channels=atom_hidden_channels, **atom_radius_graph["kwargs"])


    def forward(self, batch_data):
        atom_fea = self.atom_emb(batch_data["atom_radius_graph"]["numbers"])
        batch_data["atom_radius_graph"]["v"] = atom_fea
        atom_fea = self.atom_radius_graph(batch_data["atom_radius_graph"])
        # 读出
        batch_data["atom_radius_graph_readout"] = scatter(atom_fea, batch_data["atom_radius_graph"]["batch"], dim=0, reduce=self.reduce)
        