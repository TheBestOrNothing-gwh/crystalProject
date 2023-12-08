import torch.nn as nn
from torch_scatter import scatter

from crystalproject.utils.registry import registry
from crystalproject.module.utils.atom_embedding import AtomEmbedding


# 完全体拓扑网络
@registry.register_model("toponet")
class TopoNet(nn.Module):
    """
    完整的拓扑网络, 分为三层图神经网络, 第1,3层使用的网络是3D图神经网络, 或者是schnet、dimenet++、spherenet, 
    第二层则是一个简单的图卷积网络
    """

    def __init__(
        self,
        atom_embedding={},
        atom_hidden_channels=128,
        atom_graph={},
        atom_cluster_reduce="sum",
        cluster_hidden_channels=256,
        cluster_graph={},
        cluster_network_reduce="sum",
        underling_network={},
        reduce="mean",
    ):
        """
        Initialize TopoNet
        """
        super(TopoNet, self).__init__()
        self.atom_cluster_reduce = atom_cluster_reduce
        self.cluster_network_reduce = cluster_network_reduce
        self.reduce = reduce
        self.atom_emb = AtomEmbedding(**atom_embedding)
        self.embedding_atom = nn.Linear(
            self.atom_emb.get_dim(), atom_hidden_channels, bias=False
        )
        self.embedding_cluster = nn.Linear(
            atom_hidden_channels, cluster_hidden_channels, bias=False
        )
        model_cls = registry.get_model_class(atom_graph["name"])
        self.atom_graph = model_cls(hidden_channels=atom_hidden_channels, **atom_graph["kwargs"])
        model_cls = registry.get_model_class(cluster_graph["name"])
        self.cluster_graph = model_cls(hidden_channels=cluster_hidden_channels, **cluster_graph["kwargs"])
        model_cls = registry.get_model_class(underling_network["name"])
        self.underling_network = model_cls(hidden_channels=cluster_hidden_channels, **underling_network["kwargs"])
    
    def forward(self, batch_data):
        """
        先做原子的初始嵌入，然后再输入到原子图中，然后得到了池化得到原子簇的表示，再输入到粗粒度图中，最后再传入到底层网络中
        """
        # 原子嵌入
        atom_fea = self.atom_emb(batch_data["atom_radius_graph"]["numbers"])
        atom_fea = self.embedding_atom(atom_fea)
        batch_data["atom_radius_graph"]["v"] = atom_fea
        atom_fea = self.atom_graph(batch_data["atom_radius_graph"])
        # 读出得到原子簇的表示
        inter = batch_data["cluster_graph"]["inter"]
        cluster_fea = scatter(atom_fea[inter[0]], inter[1], dim=0, reduce=self.atom_cluster_reduce)
        cluster_fea = self.embedding_cluster(cluster_fea)
        batch_data["cluster_graph"]["v"] = cluster_fea
        cluster_fea = self.cluster_graph(batch_data["cluster_graph"])
        # 读出底层网络节点的表示
        inter = batch_data["underling_network"]["inter"]
        network_fea = scatter(cluster_fea[inter[0]], inter[1], dim=0, reduce=self.cluster_network_reduce)
        batch_data["underling_network"]["v"] = network_fea
        network_fea = self.underling_network(batch_data["underling_network"])
        # 读出完全的表示
        out = scatter(network_fea, batch_data["batch"]["network"], dim=0, reduce=self.reduce)
        out = self.embedding_cluster(out)
        return out
