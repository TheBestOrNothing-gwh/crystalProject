import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from crystalproject.utils.registry import registry


@registry.register_dataset("CrystalTopoDataset")
class CrystalTopoDataset(Dataset):
    def __init__(self, root_dir, stage="predict", target_index=[0]):
        match stage:
            case "train" | "val" | "test":
                self.root_dir = os.path.join(root_dir, stage)
            case _:
                self.root_dir = root_dir
        with open(os.path.join(self.root_dir, "id_prop.json"), "r") as f:
            self.id_prop_data = json.load(f)
            self.ids = list(self.id_prop_data.keys())
        self.target_index = target_index
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        data = np.load(os.path.join(self.root_dir, id+'.pkl'))

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

    @staticmethod
    def collate(dataset_list):
        batch_data = {}
        (
            (
                batch_data["atom_graph"]["number"],
                batch_data["atom_graph"]["edges"],
                batch_data["atom_graph"]["pos"],
                batch_data["atom_graph"]["offsets_real"],
                batch_data["atom_graph"]["offsets"],
                batch_data["atom_graph"]["rvecs"],
            ),
            (
                batch_data["cluster_graph"]["indices"],
                batch_data["cluster_graph"]["edges"],
                batch_data["cluster_graph"]["pos"],
                batch_data["cluster_graph"]["offsets_real"],
                batch_data["cluster_graph"]["offsets"],
                batch_data["cluster_graph"]["rvecs"],
            ),
            (
                batch_data["underling_network"]["indices"],
                batch_data["underling_network"]["edges"],
                batch_data["underling_network"]["pos"],
                batch_data["underling_network"]["offsets_real"],
                batch_data["underling_network"]["offsets"],
                batch_data["underling_network"]["rvecs"],
            ),
            batch_data["target"],
        ) = (([], [], [], [], []), ([], [], [], [], []), ([], [], [], [], []), [])
        base_atom_idx = 0
        base_cluster_idx = 0
        for _, data in enumerate(dataset_list):
            n_atom = data["atom_graph"]["number"].shape[0]
            n_cluster = len(data["cluster_graph"]["indices"])
            
            batch_data["atom_graph"]["number"].append(data["atom_graph"]["number"])
            batch_data["atom_graph"]["edges"].append(data["atom_graph"]["edges"] + base_atom_idx)
            batch_data["atom_graph"]["pos"].append(data["atom_graph"]["pos"])
            batch_data["atom_graph"]["offsets_real"].append(data["atom_graph"]["offsets_real"])
            batch_data["atom_graph"]["offsets"].append(data["atom_graph"]["offsets"])
            batch_data["atom_graph"]["rvecs"].append(data["atom_graph"]["rvecs"])

            batch_data["cluster_graph"]["indices"].extend([indice + base_atom_idx for indice in data["cluster_graph"]["indices"]])
            batch_data["cluster_graph"]["edges"].append(data["cluster_graph"]["edges"] + base_cluster_idx)
            batch_data["cluster_graph"]["pos"].append(data["cluster_graph"]["pos"])
            batch_data["cluster_graph"]["offsets_real"].append(data["cluster_graph"]["offsets_real"])
            batch_data["cluster_graph"]["offsets"].append(data["cluster_graph"]["offsets"])
            batch_data["cluster_graph"]["rvecs"].append(data["cluster_graph"]["rvecs"])

            batch_data["underling_network"]["indices"].extend([indice + base_cluster_idx for indice in data["underling_network"]["indices"]])
            batch_data["underling_network"]["edges"].append(data["underling_network"]["edges"] + base_cluster_idx)
            batch_data["underling_network"]["pos"].append(data["underling_network"]["pos"])
            batch_data["underling_network"]["offsets_real"].append(data["underling_network"]["offsets_real"])
            batch_data["underling_network"]["offsets"].append(data["underling_network"]["offsets"])
            batch_data["underling_network"]["rvecs"].append(data["underling_network"]["rvecs"])

            batch_data["target"].append(data["target"])

            base_atom_idx += n_atom
            base_cluster_idx += n_cluster
        
        batch_data["atom_graph"]["number"] = torch.cat(batch_data["atom_graph"]["number"], dim=0)
        batch_data["atom_graph"]["edges"] = torch.cat(batch_data["atom_graph"]["edges"], dim=1)
        batch_data["atom_graph"]["pos"] = torch.cat(batch_data["atom_graph"]["pos"], dim=0)
        batch_data["atom_graph"]["offsets_real"] = torch.cat(batch_data["atom_graph"]["offsets_real"], dim=0)
        batch_data["atom_graph"]["offsets"] = torch.cat(batch_data["atom_graph"]["offsets"], dim=0)
        
        batch_data["cluster_graph"]["edges"] = torch.cat(batch_data["cluster_graph"]["edges"], dim=1)
        batch_data["cluster_graph"]["pos"] = torch.cat(batch_data["cluster_graph"]["pos"], dim=0)
        batch_data["cluster_graph"]["offsets_real"] = torch.cat(batch_data["cluster_graph"]["offsets_real"], dim=0)
        batch_data["cluster_graph"]["offsets"] = torch.cat(batch_data["cluster_graph"]["offsets"], dim=0)

        batch_data["underling_network"]["edges"] = torch.cat(batch_data["underling_network"]["edges"], dim=1)
        batch_data["underling_network"]["pos"] = torch.cat(batch_data["underling_network"]["pos"], dim=0)
        batch_data["underling_network"]["offsets_real"] = torch.cat(batch_data["underling_network"]["offsets_real"], dim=0)
        batch_data["underling_network"]["offsets"] = torch.cat(batch_data["underling_network"]["offsets"], dim=0)

        batch_data["target"] = torch.cat(batch_data["target"], dim=0)

        return batch_data