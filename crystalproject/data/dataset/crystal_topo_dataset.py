import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from crystalproject.utils.registry import registry
from crystalproject.data.prepare.crystal_topo import create_crystal_topo


@registry.register_dataset("CrystalTopoDataset")
class CrystalTopoDataset(Dataset):
    def __init__(self, root_dir, stage="predict", target_index=["name"], on_the_fly=False, radius=8.0, max_nbr_num=12):
        match stage:
            case "train" | "val" | "test":
                self.root_dir = os.path.join(root_dir, stage)
            case _:
                self.root_dir = root_dir
        self.target_index = target_index
        self.datas = pd.read_csv(os.path.join(self.root_dir, "id_prop.csv"), names=self.target_index)
        self.on_the_fly = on_the_fly
        self.radius = radius
        self.max_nbr_num = max_nbr_num
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        name = self.datas[index]["name"]
        if self.on_the_fly:
            data = create_crystal_topo(os.path.join(self.root_dir, name+".cif"), self.radius, self.max_nbr_num)
        else:
            data = np.load(os.path.join(self.root_dir, name+'.pkl'))

        data["atom_radius_graph"]["number"] = torch.tensor(data["atom_radius_graph"]["number"], dtype=torch.int32)
        data["atom_radius_graph"]["edges"] = torch.tensor(data["atom_radius_graph"]["edges"], dtype=torch.int64)
        data["atom_radius_graph"]["pos"] = torch.tensor(data["atom_radius_graph"]["pos"], dtype=torch.float32)
        data["atom_radius_graph"]["offsets_real"] = torch.tensor(data["atom_radius_graph"]["offsets_real"], dtpe=torch.float32)
        data["atom_radius_graph"]["offsets"] = torch.tensor(data["atom_radius_graph"]["offsets"], dtpe=torch.int32)
        data["atom_radius_graph"]["rvesc"] = torch.tensor(data["atom_radius_graph"]["rvecs"], dtype=torch.float32)

        data["atom_graph"]["number"] = torch.tensor(data["atom_graph"]["number"], dtype=torch.int32)
        data["atom_graph"]["edges"] = torch.tensor(data["atom_graph"]["edges"], dtype=torch.int64)
        data["atom_graph"]["pos"] = torch.tensor(data["atom_graph"]["pos"], dtype=torch.float32)
        data["atom_graph"]["offsets_real"] = torch.tensor(data["atom_graph"]["offsets_real"], dtpe=torch.float32)
        data["atom_graph"]["offsets"] = torch.tensor(data["atom_graph"]["offsets"], dtpe=torch.int32)
        data["atom_graph"]["rvesc"] = torch.tensor(data["atom_graph"]["rvecs"], dtype=torch.float32)

        data["cluster_graph"]["inter"] = torch.tensor(data["cluster_graph"]["inter"], dtype=torch.int32)
        data["cluster_graph"]["edges"] = torch.tensor(data["cluster_graph"]["edges"], dtype=torch.int64)
        data["cluster_graph"]["pos"] = torch.tensor(data["cluster_graph"]["pos"], dtype=torch.float32)
        data["cluster_graph"]["offsets_real"] = torch.tensor(data["cluster_graph"]["offsets_real"], dtpe=torch.float32)
        data["cluster_graph"]["offsets"] = torch.tensor(data["cluster_graph"]["offsets"], dtype=torch.int32)
        data["cluster_graph"]["rvecs"] = torch.tensor(data["cluster_graph"]["rvecs"], dtype=torch.float32)

        data["underling_network"]["inter"] = torch.tensor(data["underling_network"]["inter"], dtype=torch.int32)
        data["underling_network"]["edges"] = torch.tensor(data["underling_network"]["edges"], dtype=torch.int64)
        data["underling_network"]["pos"] = torch.tensor(data["underling_network"]["pos"], dtype=torch.float32)
        data["underling_network"]["offsets_real"] = torch.tensor(data["underling_network"]["offsets_real"], dtpe=torch.float32)
        data["underling_network"]["offsets"] = torch.tensor(data["underling_network"]["offsets"], dtype=torch.int32)
        data["underling_network"]["rvecs"] = torch.tensor(data["underling_network"]["rvecs"], dtype=torch.float32)

        data["target"] = torch.tensor([[self.datas[index][i] for i in self.target_index if i != "name"]], dtype=torch.float32)

        return data

    @staticmethod
    def collate(dataset_list):
        batch_data = {}
        (
            (
                batch_data["atom_radius_graph"]["number"],
                batch_data["atom_radius_graph"]["edges"],
                batch_data["atom_radius_graph"]["pos"],
                batch_data["atom_radius_graph"]["offsets_real"],
                batch_data["atom_radius_graph"]["offsets"],
                batch_data["atom_radius_graph"]["rvecs"],
            ),
            (
                batch_data["atom_graph"]["number"],
                batch_data["atom_graph"]["edges"],
                batch_data["atom_graph"]["pos"],
                batch_data["atom_graph"]["offsets_real"],
                batch_data["atom_graph"]["offsets"],
                batch_data["atom_graph"]["rvecs"],
            ),
            (
                batch_data["cluster_graph"]["inter"],
                batch_data["cluster_graph"]["edges"],
                batch_data["cluster_graph"]["pos"],
                batch_data["cluster_graph"]["offsets_real"],
                batch_data["cluster_graph"]["offsets"],
                batch_data["cluster_graph"]["rvecs"],
            ),
            (
                batch_data["underling_network"]["inter"],
                batch_data["underling_network"]["edges"],
                batch_data["underling_network"]["pos"],
                batch_data["underling_network"]["offsets_real"],
                batch_data["underling_network"]["offsets"],
                batch_data["underling_network"]["rvecs"],
            ),
            batch_data["batch"],
            batch_data["target"],
        ) = (([], [], [], [], []), ([], [], [], [], []), ([], [], [], [], []), [], [])
        base_atom_idx = 0
        base_cluster_idx = 0
        base_network_idx = 0
        for i, data in enumerate(dataset_list):
            n_atom = data["atom_graph"]["pos"].shape[0]
            n_cluster = len(data["cluster_graph"]["pos"]).shape[0]
            n_network = len(data["underling_network"]["pos"]).shape[0]

            batch_data["atom_radius_graph"]["number"].append(data["atom_radius_graph"]["number"])
            batch_data["atom_radius_graph"]["edges"].append(data["atom_radius_graph"]["edges"] + base_atom_idx)
            batch_data["atom_radius_graph"]["pos"].append(data["atom_radius_graph"]["pos"])
            batch_data["atom_radius_graph"]["offsets_real"].append(data["atom_radius_graph"]["offsets_real"])
            batch_data["atom_radius_graph"]["offsets"].append(data["atom_radius_graph"]["offsets"])
            batch_data["atom_radius_graph"]["rvecs"].append(data["atom_radius_graph"]["rvecs"])

            batch_data["atom_graph"]["number"].append(data["atom_graph"]["number"])
            batch_data["atom_graph"]["edges"].append(data["atom_graph"]["edges"] + base_atom_idx)
            batch_data["atom_graph"]["pos"].append(data["atom_graph"]["pos"])
            batch_data["atom_graph"]["offsets_real"].append(data["atom_graph"]["offsets_real"])
            batch_data["atom_graph"]["offsets"].append(data["atom_graph"]["offsets"])
            batch_data["atom_graph"]["rvecs"].append(data["atom_graph"]["rvecs"])

            batch_data["cluster_graph"]["inter"].extend(torch.tensor([data["cluster_graph"]["inter"][0] + base_atom_idx, data["cluster_graph"]["inter"][1] + base_cluster_idx]))
            batch_data["cluster_graph"]["edges"].append(data["cluster_graph"]["edges"] + base_cluster_idx)
            batch_data["cluster_graph"]["pos"].append(data["cluster_graph"]["pos"])
            batch_data["cluster_graph"]["offsets_real"].append(data["cluster_graph"]["offsets_real"])
            batch_data["cluster_graph"]["offsets"].append(data["cluster_graph"]["offsets"])
            batch_data["cluster_graph"]["rvecs"].append(data["cluster_graph"]["rvecs"])

            batch_data["underling_network"]["inter"].extend(torch.tensor([data["underling_network"]["inter"][0] + base_cluster_idx, data["underling_network"]["inter"][1] + base_network_idx]))
            batch_data["underling_network"]["edges"].append(data["underling_network"]["edges"] + base_network_idx)
            batch_data["underling_network"]["pos"].append(data["underling_network"]["pos"])
            batch_data["underling_network"]["offsets_real"].append(data["underling_network"]["offsets_real"])
            batch_data["underling_network"]["offsets"].append(data["underling_network"]["offsets"])
            batch_data["underling_network"]["rvecs"].append(data["underling_network"]["rvecs"])

            batch_data["batch"].append(torch.full(n_network, i))
            batch_data["target"].append(data["target"])

            base_atom_idx += n_atom
            base_cluster_idx += n_cluster
            base_network_idx += n_network
        
        batch_data["atom_radius_graph"]["number"] = torch.cat(batch_data["atom_radius_graph"]["number"], dim=0)
        batch_data["atom_radius_graph"]["edges"] = torch.cat(batch_data["atom_radius_graph"]["edges"], dim=1)
        batch_data["atom_radius_graph"]["pos"] = torch.cat(batch_data["atom_radius_graph"]["pos"], dim=0)
        batch_data["atom_radius_graph"]["offsets_real"] = torch.cat(batch_data["atom_radius_graph"]["offsets_real"], dim=0)
        batch_data["atom_radius_graph"]["offsets"] = torch.cat(batch_data["atom_radius_graph"]["offsets"], dim=0)

        batch_data["atom_graph"]["number"] = torch.cat(batch_data["atom_graph"]["number"], dim=0)
        batch_data["atom_graph"]["edges"] = torch.cat(batch_data["atom_graph"]["edges"], dim=1)
        batch_data["atom_graph"]["pos"] = torch.cat(batch_data["atom_graph"]["pos"], dim=0)
        batch_data["atom_graph"]["offsets_real"] = torch.cat(batch_data["atom_graph"]["offsets_real"], dim=0)
        batch_data["atom_graph"]["offsets"] = torch.cat(batch_data["atom_graph"]["offsets"], dim=0)
        
        batch_data["cluster_graph"]["inter"] = torch.cat(batch_data["cluster_graph"]["inter"], dim=1)
        batch_data["cluster_graph"]["edges"] = torch.cat(batch_data["cluster_graph"]["edges"], dim=1)
        batch_data["cluster_graph"]["pos"] = torch.cat(batch_data["cluster_graph"]["pos"], dim=0)
        batch_data["cluster_graph"]["offsets_real"] = torch.cat(batch_data["cluster_graph"]["offsets_real"], dim=0)
        batch_data["cluster_graph"]["offsets"] = torch.cat(batch_data["cluster_graph"]["offsets"], dim=0)

        batch_data["underling_network"]["inter"] = torch.cat(batch_data["underling_network"]["inter"], dim=1)
        batch_data["underling_network"]["edges"] = torch.cat(batch_data["underling_network"]["edges"], dim=1)
        batch_data["underling_network"]["pos"] = torch.cat(batch_data["underling_network"]["pos"], dim=0)
        batch_data["underling_network"]["offsets_real"] = torch.cat(batch_data["underling_network"]["offsets_real"], dim=0)
        batch_data["underling_network"]["offsets"] = torch.cat(batch_data["underling_network"]["offsets"], dim=0)

        batch_data["batch"] = torch.cat(batch_data["batch"], dim=0)
        batch_data["target"] = torch.cat(batch_data["target"], dim=0)

        return batch_data