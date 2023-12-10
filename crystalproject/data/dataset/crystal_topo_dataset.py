import torch
import os
import pandas as pd
import json
import pickle
from torch.utils.data import Dataset

from crystalproject.utils.registry import registry
from crystalproject.data.prepare.process.crystal_topo import create_crystal_topo


@registry.register_dataset("CrystalTopoDataset")
class CrystalTopoDataset(Dataset):
    def __init__(self, root_dir, stage="predict", descriptor_index=[], target_index=[], on_the_fly=False, radius=5.0, max_nbr_num=12):
        self.root_dir = root_dir
        match stage:
            case "train" | "val" | "test":
                self.id_prop = "id_prop_" + stage + ".json"
            case "predict":
                self.id_prop = "id_prop_all.json"
            case _:
                pass
        self.descriptor_index = descriptor_index
        self.target_index = target_index
        with open(os.path.join(self.root_dir, self.id_prop)) as f:
            datas = json.load(f)
        self.datas = pd.json_normalize(datas)
        self.on_the_fly = on_the_fly
        self.radius = radius
        self.max_nbr_num = max_nbr_num
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        value = self.datas.loc[index]
        name = value["name"]
        if self.on_the_fly:
            data = create_crystal_topo(os.path.join(self.root_dir, "all", name+".cif"), self.radius, self.max_nbr_num, value["use_bond_types"], value["bond_types"], value["linker_types"])
        else:
            data = pickle.load(open(os.path.join(self.root_dir, "all", name+'.pkl'), "rb"))

        data["atom_radius_graph"]["numbers"] = torch.tensor(data["atom_radius_graph"]["numbers"], dtype=torch.int32)
        data["atom_radius_graph"]["edges"] = torch.tensor(data["atom_radius_graph"]["edges"], dtype=torch.int64)
        data["atom_radius_graph"]["pos"] = torch.tensor(data["atom_radius_graph"]["pos"], dtype=torch.float32)
        data["atom_radius_graph"]["offsets_real"] = torch.tensor(data["atom_radius_graph"]["offsets_real"], dtype=torch.float32)
        data["atom_radius_graph"]["offsets"] = torch.tensor(data["atom_radius_graph"]["offsets"], dtype=torch.int32)
        data["atom_radius_graph"]["rvesc"] = torch.tensor(data["atom_radius_graph"]["rvecs"], dtype=torch.float32)

        data["atom_graph"]["numbers"] = torch.tensor(data["atom_graph"]["numbers"], dtype=torch.int32)
        data["atom_graph"]["edges"] = torch.tensor(data["atom_graph"]["edges"], dtype=torch.int64)
        data["atom_graph"]["pos"] = torch.tensor(data["atom_graph"]["pos"], dtype=torch.float32)
        data["atom_graph"]["offsets_real"] = torch.tensor(data["atom_graph"]["offsets_real"], dtype=torch.float32)
        data["atom_graph"]["offsets"] = torch.tensor(data["atom_graph"]["offsets"], dtype=torch.int32)
        data["atom_graph"]["rvesc"] = torch.tensor(data["atom_graph"]["rvecs"], dtype=torch.float32)

        data["cluster_graph"]["inter"] = torch.tensor(data["cluster_graph"]["inter"], dtype=torch.int32)
        data["cluster_graph"]["edges"] = torch.tensor(data["cluster_graph"]["edges"], dtype=torch.int64)
        data["cluster_graph"]["pos"] = torch.tensor(data["cluster_graph"]["pos"], dtype=torch.float32)
        data["cluster_graph"]["offsets_real"] = torch.tensor(data["cluster_graph"]["offsets_real"], dtype=torch.float32)
        data["cluster_graph"]["offsets"] = torch.tensor(data["cluster_graph"]["offsets"], dtype=torch.int32)
        data["cluster_graph"]["rvecs"] = torch.tensor(data["cluster_graph"]["rvecs"], dtype=torch.float32)

        data["underling_network"]["inter"] = torch.tensor(data["underling_network"]["inter"], dtype=torch.int32)
        data["underling_network"]["edges"] = torch.tensor(data["underling_network"]["edges"], dtype=torch.int64)
        data["underling_network"]["pos"] = torch.tensor(data["underling_network"]["pos"], dtype=torch.float32)
        data["underling_network"]["offsets_real"] = torch.tensor(data["underling_network"]["offsets_real"], dtype=torch.float32)
        data["underling_network"]["offsets"] = torch.tensor(data["underling_network"]["offsets"], dtype=torch.int32)
        data["underling_network"]["rvecs"] = torch.tensor(data["underling_network"]["rvecs"], dtype=torch.float32)

        data["descriptor"] = torch.cat([torch.tensor(value[i], dtype=torch.float32) if value[i].size > 1 else torch.tensor([value[i]], dtype=torch.float32) for i in self.descriptor_index], dim=0).unsqueeze(dim=0)
        data["target"] = torch.cat([torch.tensor(value[i], dtype=torch.float32) if value[i].size > 1 else torch.tensor([value[i]], dtype=torch.float32) for i in self.target_index], dim=0).unsqueeze(dim=0)

        return data

    @staticmethod
    def collate(dataset_list):
        batch_data = {
            "atom_radius_graph": {},
            "atom_graph": {},
            "cluster_graph": {},
            "underling_network": {},
            "batch": {}
        }
        (
            (
                batch_data["atom_radius_graph"]["numbers"],
                batch_data["atom_radius_graph"]["edges"],
                batch_data["atom_radius_graph"]["edges_devide"],
                batch_data["atom_radius_graph"]["pos"],
                batch_data["atom_radius_graph"]["offsets_real"],
                batch_data["atom_radius_graph"]["offsets"],
                batch_data["atom_radius_graph"]["rvecs"],
            ),
            (
                batch_data["atom_graph"]["numbers"],
                batch_data["atom_graph"]["edges"],
                batch_data["atom_graph"]["edges_devide"],
                batch_data["atom_graph"]["pos"],
                batch_data["atom_graph"]["offsets_real"],
                batch_data["atom_graph"]["offsets"],
                batch_data["atom_graph"]["rvecs"],
            ),
            (
                batch_data["cluster_graph"]["inter"],
                batch_data["cluster_graph"]["edges"],
                batch_data["cluster_graph"]["edges_devide"],
                batch_data["cluster_graph"]["pos"],
                batch_data["cluster_graph"]["offsets_real"],
                batch_data["cluster_graph"]["offsets"],
                batch_data["cluster_graph"]["rvecs"],
            ),
            (
                batch_data["underling_network"]["inter"],
                batch_data["underling_network"]["edges"],
                batch_data["underling_network"]["edges_devide"],
                batch_data["underling_network"]["pos"],
                batch_data["underling_network"]["offsets_real"],
                batch_data["underling_network"]["offsets"],
                batch_data["underling_network"]["rvecs"],
            ),
            (
                batch_data["batch"]["atom"],
                batch_data["batch"]["cluster"],
                batch_data["batch"]["network"]
            ),
            batch_data["descriptor"],
            batch_data["target"],
        ) = (([], [], [], [], [], [], []), ([], [], [], [], [], [], []), ([], [], [], [], [], [], []), ([], [], [], [], [], [], []), ([], [], []), [], [])
        base_atom_idx = 0
        base_cluster_idx = 0
        base_network_idx = 0
        for i, data in enumerate(dataset_list):
            n_atom = data["atom_graph"]["pos"].shape[0]
            n_cluster = data["cluster_graph"]["pos"].shape[0]
            n_network = data["underling_network"]["pos"].shape[0]

            batch_data["atom_radius_graph"]["numbers"].append(data["atom_radius_graph"]["numbers"])
            batch_data["atom_radius_graph"]["edges"].append(data["atom_radius_graph"]["edges"] + base_atom_idx)
            batch_data["atom_radius_graph"]["edges_devide"].append(data["atom_radius_graph"]["edges"].shape[1])
            batch_data["atom_radius_graph"]["pos"].append(data["atom_radius_graph"]["pos"])
            batch_data["atom_radius_graph"]["offsets_real"].append(data["atom_radius_graph"]["offsets_real"])
            batch_data["atom_radius_graph"]["offsets"].append(data["atom_radius_graph"]["offsets"])
            batch_data["atom_radius_graph"]["rvecs"].append(data["atom_radius_graph"]["rvecs"])

            batch_data["atom_graph"]["numbers"].append(data["atom_graph"]["numbers"])
            batch_data["atom_graph"]["edges"].append(data["atom_graph"]["edges"] + base_atom_idx)
            batch_data["atom_graph"]["edges_devide"].append(data["atom_graph"]["edges"].shape[1])
            batch_data["atom_graph"]["pos"].append(data["atom_graph"]["pos"])
            batch_data["atom_graph"]["offsets_real"].append(data["atom_graph"]["offsets_real"])
            batch_data["atom_graph"]["offsets"].append(data["atom_graph"]["offsets"])
            batch_data["atom_graph"]["rvecs"].append(data["atom_graph"]["rvecs"])

            batch_data["cluster_graph"]["inter"].append(data["cluster_graph"]["inter"] + torch.tensor([[base_atom_idx], [base_cluster_idx]]))
            batch_data["cluster_graph"]["edges"].append(data["cluster_graph"]["edges"] + base_cluster_idx)
            batch_data["cluster_graph"]["edges_devide"].append(data["cluster_graph"]["edges"].shape[1])
            batch_data["cluster_graph"]["pos"].append(data["cluster_graph"]["pos"])
            batch_data["cluster_graph"]["offsets_real"].append(data["cluster_graph"]["offsets_real"])
            batch_data["cluster_graph"]["offsets"].append(data["cluster_graph"]["offsets"])
            batch_data["cluster_graph"]["rvecs"].append(data["cluster_graph"]["rvecs"])

            batch_data["underling_network"]["inter"].append(data["underling_network"]["inter"] + torch.tensor([[base_cluster_idx], [base_network_idx]]))
            batch_data["underling_network"]["edges"].append(data["underling_network"]["edges"] + base_network_idx)
            batch_data["underling_network"]["edges_devide"].append(data["underling_network"]["edges"].shape[1])
            batch_data["underling_network"]["pos"].append(data["underling_network"]["pos"])
            batch_data["underling_network"]["offsets_real"].append(data["underling_network"]["offsets_real"])
            batch_data["underling_network"]["offsets"].append(data["underling_network"]["offsets"])
            batch_data["underling_network"]["rvecs"].append(data["underling_network"]["rvecs"])

            batch_data["batch"]["atom"].append(torch.full((n_atom,), i))
            batch_data["batch"]["cluster"].append(torch.full((n_cluster,), i))
            batch_data["batch"]["network"].append(torch.full((n_network,), i))
            
            batch_data["descriptor"].append(data["descriptor"])
            batch_data["target"].append(data["target"])

            base_atom_idx += n_atom
            base_cluster_idx += n_cluster
            base_network_idx += n_network
        
        batch_data["atom_radius_graph"]["numbers"] = torch.cat(batch_data["atom_radius_graph"]["numbers"], dim=0)
        batch_data["atom_radius_graph"]["edges"] = torch.cat(batch_data["atom_radius_graph"]["edges"], dim=1)
        batch_data["atom_radius_graph"]["edges_devide"] = torch.tensor(batch_data["atom_radius_graph"]["edges_devide"], dtype=torch.int32)
        batch_data["atom_radius_graph"]["pos"] = torch.cat(batch_data["atom_radius_graph"]["pos"], dim=0)
        batch_data["atom_radius_graph"]["offsets_real"] = torch.cat(batch_data["atom_radius_graph"]["offsets_real"], dim=0)
        batch_data["atom_radius_graph"]["offsets"] = torch.cat(batch_data["atom_radius_graph"]["offsets"], dim=0)

        batch_data["atom_graph"]["numbers"] = torch.cat(batch_data["atom_graph"]["numbers"], dim=0)
        batch_data["atom_graph"]["edges"] = torch.cat(batch_data["atom_graph"]["edges"], dim=1)
        batch_data["atom_graph"]["edges_devide"] = torch.tensor(batch_data["atom_graph"]["edges_devide"], dtype=torch.int32)
        batch_data["atom_graph"]["pos"] = torch.cat(batch_data["atom_graph"]["pos"], dim=0)
        batch_data["atom_graph"]["offsets_real"] = torch.cat(batch_data["atom_graph"]["offsets_real"], dim=0)
        batch_data["atom_graph"]["offsets"] = torch.cat(batch_data["atom_graph"]["offsets"], dim=0)
        
        batch_data["cluster_graph"]["inter"] = torch.cat(batch_data["cluster_graph"]["inter"], dim=1)
        batch_data["cluster_graph"]["edges"] = torch.cat(batch_data["cluster_graph"]["edges"], dim=1)
        batch_data["cluster_graph"]["edges_devide"] = torch.tensor(batch_data["cluster_graph"]["edges_devide"], dtype=torch.int32)
        batch_data["cluster_graph"]["pos"] = torch.cat(batch_data["cluster_graph"]["pos"], dim=0)
        batch_data["cluster_graph"]["offsets_real"] = torch.cat(batch_data["cluster_graph"]["offsets_real"], dim=0)
        batch_data["cluster_graph"]["offsets"] = torch.cat(batch_data["cluster_graph"]["offsets"], dim=0)

        batch_data["underling_network"]["inter"] = torch.cat(batch_data["underling_network"]["inter"], dim=1)
        batch_data["underling_network"]["edges"] = torch.cat(batch_data["underling_network"]["edges"], dim=1)
        batch_data["underling_network"]["edges_devide"] = torch.tensor(batch_data["underling_network"]["edges_devide"], dtype=torch.int32)
        batch_data["underling_network"]["pos"] = torch.cat(batch_data["underling_network"]["pos"], dim=0)
        batch_data["underling_network"]["offsets_real"] = torch.cat(batch_data["underling_network"]["offsets_real"], dim=0)
        batch_data["underling_network"]["offsets"] = torch.cat(batch_data["underling_network"]["offsets"], dim=0)

        batch_data["batch"]["atom"] = torch.cat(batch_data["batch"]["atom"], dim=0)
        batch_data["batch"]["cluster"] = torch.cat(batch_data["batch"]["cluster"], dim=0)
        batch_data["batch"]["network"] = torch.cat(batch_data["batch"]["network"], dim=0)

        batch_data["descriptor"] = torch.cat(batch_data["descriptor"], dim=0)
        batch_data["target"] = torch.cat(batch_data["target"], dim=0)
        return batch_data