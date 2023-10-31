import os
import torch
from crystalproject.utils.registry import registry
from crystalproject.data.dataset.crystal_topo_dataset import CrystalTopoDataset
from crystalproject.data.prepare.crystal_topo import create_crystal_topo


@registry.register_dataset("CrystalTopoDatasetOnTheFly")
class CrystalTopoDatasetOnTheFly(CrystalTopoDataset):
    def __init__(self, root_dir, stage="predict", target_index=[0]):
        super(CrystalTopoDataset, self).__init__(root_dir, stage, target_index)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]
        data = create_crystal_topo(os.path.join(self.root_dir, id+".cif"))

        data["atom_graph"]["number"] = torch.tensor(data["atom_graph"]["number"], dtype=torch.int32)
        data["atom_graph"]["edges"] = torch.tensor(data["atom_graph"]["edges"], dtype=torch.int64)
        data["atom_graph"]["pos"] = torch.tensor(data["atom_graph"]["pos"], dtype=torch.float32)
        data["atom_graph"]["offsets_real"] = torch.tensor(data["atom_graph"]["offsets_real"], dtpe=torch.float32)
        data["atom_graph"]["offsets"] = torch.tensor(data["atom_graph"]["offsets"], dtpe=torch.int32)
        data["atom_graph"]["rvesc"] = torch.tensor(data["atom_graph"]["rvecs"], dtype=torch.float32)

        data["cluster_graph"]["indices"] = torch.tensor(data["cluster_graph"]["indices"], dtype=torch.int32)
        data["cluster_graph"]["edges"] = torch.tensor(data["cluster_graph"]["edges"], dtype=torch.int64)
        data["cluster_graph"]["pos"] = torch.tensor(data["cluster_graph"]["pos"], dtype=torch.float32)
        data["cluster_graph"]["offsets_real"] = torch.tensor(data["cluster_graph"]["offsets_real"], dtpe=torch.float32)
        data["cluster_graph"]["offsets"] = torch.tensor(data["cluster_graph"]["offsets"], dtype=torch.int32)
        data["cluster_graph"]["rvecs"] = torch.tensor(data["cluster_graph"]["rvecs"], dtype=torch.float32)

        data["underling_network"]["indices"] = torch.tensor(data["underling_network"]["indices"], dtype=torch.int32)
        data["underling_network"]["edges"] = torch.tensor(data["underling_network"]["edges"], dtype=torch.int64)
        data["underling_network"]["pos"] = torch.tensor(data["underling_network"]["pos"], dtype=torch.float32)
        data["underling_network"]["offsets_real"] = torch.tensor(data["underling_network"]["offsets_real"], dtpe=torch.float32)
        data["underling_network"]["offsets"] = torch.tensor(data["underling_network"]["offsets"], dtype=torch.int32)
        data["underling_network"]["rvecs"] = torch.tensor(data["underling_network"]["rvecs"], dtype=torch.float32)

        data["target"] = torch.tensor([[self.id_prop_data[id][i] for i in self.target_index]], dtype=torch.float32)

        return data
    