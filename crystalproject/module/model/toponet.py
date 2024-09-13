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
        atom_radius_graph={},
        atom_bond_graph={},
        atom_cluster_reduce="max",
        cluster_hidden_channels=128,
        cluster_graph={},
        cluster_linker_reduce="max",
        linker_graph={},
        linker_network_reduce="max",
        underling_network={},
        reduce="mean",
    ):
        """
        Initialize TopoNet
        """
        super(TopoNet, self).__init__()
        self.atom_cluster_reduce = atom_cluster_reduce
        self.cluster_linker_reduce = cluster_linker_reduce
        self.linker_network_reduce = linker_network_reduce
        self.reduce = reduce
        self.atom_emb = AtomEmbedding(hidden_channels=atom_hidden_channels, **atom_embedding)
        self.atom_radius_graph, self.atom_bond_graph, self.cluster_graph, self.linker_graph, self.underling_network = atom_radius_graph, atom_bond_graph, cluster_graph, linker_graph, underling_network
        if self.atom_radius_graph != {}:
            model_cls = registry.get_model_class(atom_radius_graph["name"])
            self.atom_radius_graph = model_cls(hidden_channels=atom_hidden_channels, **atom_radius_graph["kwargs"])
        if self.atom_bond_graph != {}:
            model_cls = registry.get_model_class(atom_bond_graph["name"])
            self.atom_bond_graph = model_cls(hidden_channels=atom_hidden_channels, **atom_bond_graph["kwargs"])
        if self.cluster_graph != {}:
            model_cls = registry.get_model_class(cluster_graph["name"])
            self.cluster_graph = model_cls(hidden_channels=cluster_hidden_channels, **cluster_graph["kwargs"])
        if self.linker_graph != {}:
            model_cls = registry.get_model_class(linker_graph["name"])
            self.linker_graph = model_cls(hidden_channels=cluster_hidden_channels, **linker_graph["kwargs"])
        if self.underling_network != {}:
            model_cls = registry.get_model_class(underling_network["name"])
            self.underling_network = model_cls(hidden_channels=cluster_hidden_channels, **underling_network["kwargs"])
    
    def forward(self, batch_data):
        """
        先做原子的初始嵌入，然后再输入到原子图中，然后得到了池化得到原子簇的表示，再输入到粗粒度图中，最后再传入到底层网络中
        """
        # 原子嵌入(化学键版)
        if self.atom_bond_graph != {}:
            atom_bond_fea = self.atom_emb(batch_data["atom_bond_graph"]["numbers"])
            batch_data["atom_bond_graph"]["v"] = atom_bond_fea
            atom_bond_fea = self.atom_bond_graph(batch_data["atom_bond_graph"])
            # 读出原子图
            batch_data["atom_bond_graph_readout"] = scatter(atom_bond_fea, batch_data["atom_bond_graph"]["batch"], dim=0, reduce=self.reduce)

        # 原子嵌入（径向版）
        if self.atom_radius_graph != {}:
            atom_radius_fea = self.atom_emb(batch_data["atom_radius_graph"]["numbers"])
            batch_data["atom_radius_graph"]["v"] = atom_radius_fea
            atom_radius_fea = self.atom_radius_graph(batch_data["atom_radius_graph"])
            # 读出原子图
            batch_data["atom_radius_graph_readout"] = scatter(atom_radius_fea, batch_data["atom_radius_graph"]["batch"], dim=0, reduce=self.reduce)
        
        atom_fea = atom_bond_fea
        
        inter = batch_data["cluster_graph"]["inter"]
        cluster_fea = scatter(atom_fea[inter[0]], inter[1], dim=0, reduce=self.atom_cluster_reduce)
        batch_data["cluster_graph"]["v"] = cluster_fea    
        # 读出得到原子簇的表示
        if self.cluster_graph != {}:
            cluster_fea = self.cluster_graph(batch_data["cluster_graph"])
            batch_data["cluster_graph_readout"] = scatter(cluster_fea, batch_data["cluster_graph"]["batch"], dim=0, reduce=self.reduce)
        
        inter = batch_data["linker_graph"]["inter"]
        linker_fea = scatter(cluster_fea[inter[0]], inter[1], dim=0, reduce=self.cluster_linker_reduce)
        batch_data["linker_graph"]["v"] = linker_fea
        # 读出单体图的表示
        if self.linker_graph != {}:
            linker_fea = self.linker_graph(batch_data["linker_graph"])
            batch_data["linker_graph_readout"] = scatter(linker_fea, batch_data["linker_graph"]["batch"], dim=0, reduce=self.reduce)
        
        inter = batch_data["underling_network"]["inter"]
        network_fea = scatter(linker_fea[inter[0]], inter[1], dim=0, reduce=self.linker_network_reduce)
        batch_data["underling_network"]["v"] = network_fea
        # 读出底层网络节点的表示
        if self.underling_network != {}:
            network_fea = self.underling_network(batch_data["underling_network"])
            # 读出
            batch_data["underling_network_readout"] = scatter(network_fea, batch_data["underling_network"]["batch"], dim=0, reduce=self.reduce)
