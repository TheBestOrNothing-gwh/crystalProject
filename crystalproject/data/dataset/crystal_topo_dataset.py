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
    def __init__(self, input_dir, split_dir, stage="predict", descriptor_index=[], on_the_fly=False, used_topos=["atom_radius_graph"], radius=5.0, max_nbr_num=12):
        self.input_dir = input_dir
        self.split_dir = split_dir
        match stage:
            case "train" | "val" | "test":
                self.id_prop = "id_prop_" + stage + ".json"
            case "predict":
                self.id_prop = "id_prop_all.json"
            case _:
                pass
        self.used_topos = used_topos
        self.descriptor_index = descriptor_index
        with open(os.path.join(self.split_dir, self.id_prop)) as f:
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
            config = {"used_topos": self.used_topos}
            if "atom_radius_graph" in self.used_topos:
                config["atom_radius_graph"]["radius"] = self.radius
                config["atom_radius_graph"]["max_num_nbr"] = self.max_nbr_num
            if "high_order" in self.used_topos:
                config["high_order"]["use_bond_types"] = value["use_bond_types"]
                config["high_order"]["bond_types"] = value["bond_types"]
                config["high_order"]["linker_types"] = value["linker_types"]
            data = create_crystal_topo(os.path.join(self.input_dir, name+".cif"), **config)
        else:
            data = pickle.load(open(os.path.join(self.input_dir, name+'.pkl'), "rb"))       
        if "atom_radius_graph" in self.used_topos:
            data["atom_radius_graph"]["numbers"] = torch.tensor(data["atom_radius_graph"]["numbers"], dtype=torch.int32)
            data["atom_radius_graph"]["edges"] = torch.tensor(data["atom_radius_graph"]["edges"], dtype=torch.int64)
            data["atom_radius_graph"]["pos"] = torch.tensor(data["atom_radius_graph"]["pos"], dtype=torch.float32)
            data["atom_radius_graph"]["offsets_real"] = torch.tensor(data["atom_radius_graph"]["offsets_real"], dtype=torch.float32)
            data["atom_radius_graph"]["offsets"] = torch.tensor(data["atom_radius_graph"]["offsets"], dtype=torch.int32)
            data["atom_radius_graph"]["rvesc"] = torch.tensor(data["atom_radius_graph"]["rvecs"], dtype=torch.float32)
        if "atom_bond_graph" in self.used_topos:
            data["atom_bond_graph"]["numbers"] = torch.tensor(data["atom_bond_graph"]["numbers"], dtype=torch.int32)
            data["atom_bond_graph"]["edges"] = torch.tensor(data["atom_bond_graph"]["edges"], dtype=torch.int64)
            data["atom_bond_graph"]["pos"] = torch.tensor(data["atom_bond_graph"]["pos"], dtype=torch.float32)
            data["atom_bond_graph"]["offsets_real"] = torch.tensor(data["atom_bond_graph"]["offsets_real"], dtype=torch.float32)
            data["atom_bond_graph"]["offsets"] = torch.tensor(data["atom_bond_graph"]["offsets"], dtype=torch.int32)
            data["atom_bond_graph"]["rvesc"] = torch.tensor(data["atom_bond_graph"]["rvecs"], dtype=torch.float32)
        if "high_order" in self.used_topos:
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

        for descriptor in self.descriptor_index:
            data[descriptor] = torch.tensor(value[descriptor], dtype=torch.float32).unsqueeze(0) if value[descriptor].size > 1 else torch.tensor([value[descriptor]], dtype=torch.float32).unsqueeze(0)
        data["name"] = name
        return data

    def collate(self, dataset_list):
        batch_data = {
            "atom_radius_graph": {},
            "atom_bond_graph": {},
            "cluster_graph": {},
            "underling_network": {},
        }
        (
            (
                batch_data["atom_radius_graph"]["numbers"],
                batch_data["atom_radius_graph"]["edges"],
                batch_data["atom_radius_graph"]["pos"],
                batch_data["atom_radius_graph"]["offsets_real"],
                batch_data["atom_radius_graph"]["offsets"],
                batch_data["atom_radius_graph"]["rvecs"],
                batch_data["atom_radius_graph"]["batch"]
            ),
            (
                batch_data["atom_bond_graph"]["numbers"],
                batch_data["atom_bond_graph"]["edges"],
                batch_data["atom_bond_graph"]["pos"],
                batch_data["atom_bond_graph"]["offsets_real"],
                batch_data["atom_bond_graph"]["offsets"],
                batch_data["atom_bond_graph"]["rvecs"],
                batch_data["atom_bond_graph"]["batch"]
            ),
            (
                batch_data["cluster_graph"]["inter"],
                batch_data["cluster_graph"]["edges"],
                batch_data["cluster_graph"]["pos"],
                batch_data["cluster_graph"]["offsets_real"],
                batch_data["cluster_graph"]["offsets"],
                batch_data["cluster_graph"]["rvecs"],
                batch_data["cluster_graph"]["batch"]
            ),
            (
                batch_data["underling_network"]["inter"],
                batch_data["underling_network"]["edges"],
                batch_data["underling_network"]["pos"],
                batch_data["underling_network"]["offsets_real"],
                batch_data["underling_network"]["offsets"],
                batch_data["underling_network"]["rvecs"],
                batch_data["underling_network"]["batch"]
            )
        ) = (([], [], [], [], [], [], []), ([], [], [], [], [], [], []), ([], [], [], [], [], [], []), ([], [], [], [], [], [], []))
        batch_data.update(
            {descriptor: [] for descriptor in self.descriptor_index}
        )
        batch_data["name"] = []
        base_atom_radius_idx = 0
        base_atom_bond_idx = 0
        base_cluster_idx = 0
        base_network_idx = 0
        batch_data["batch_size"] = len(dataset_list)
        for i, data in enumerate(dataset_list):
            if "atom_radius_graph" in self.used_topos:
                batch_data["atom_radius_graph"]["numbers"].append(data["atom_radius_graph"]["numbers"])
                batch_data["atom_radius_graph"]["edges"].append(data["atom_radius_graph"]["edges"] + base_atom_radius_idx)
                batch_data["atom_radius_graph"]["pos"].append(data["atom_radius_graph"]["pos"])
                batch_data["atom_radius_graph"]["offsets_real"].append(data["atom_radius_graph"]["offsets_real"])
                batch_data["atom_radius_graph"]["offsets"].append(data["atom_radius_graph"]["offsets"])
                batch_data["atom_radius_graph"]["rvecs"].append(data["atom_radius_graph"]["rvecs"])
                batch_data["atom_radius_graph"]["batch"].append(torch.full((data["atom_radius_graph"]["pos"].shape[0], ), i))
            if "atom_bond_graph" in self.used_topos:
                batch_data["atom_bond_graph"]["numbers"].append(data["atom_bond_graph"]["numbers"])
                batch_data["atom_bond_graph"]["edges"].append(data["atom_bond_graph"]["edges"] + base_atom_bond_idx)
                batch_data["atom_bond_graph"]["pos"].append(data["atom_bond_graph"]["pos"])
                batch_data["atom_bond_graph"]["offsets_real"].append(data["atom_bond_graph"]["offsets_real"])
                batch_data["atom_bond_graph"]["offsets"].append(data["atom_bond_graph"]["offsets"])
                batch_data["atom_bond_graph"]["rvecs"].append(data["atom_bond_graph"]["rvecs"])
                batch_data["atom_bond_graph"]["batch"].append(torch.full((data["atom_bond_graph"]["pos"].shape[0], ), i))
            if "high_order" in self.used_topos:
                batch_data["cluster_graph"]["inter"].append(data["cluster_graph"]["inter"] + torch.tensor([[base_atom_bond_idx], [base_cluster_idx]]))
                batch_data["cluster_graph"]["edges"].append(data["cluster_graph"]["edges"] + base_cluster_idx)
                batch_data["cluster_graph"]["pos"].append(data["cluster_graph"]["pos"])
                batch_data["cluster_graph"]["offsets_real"].append(data["cluster_graph"]["offsets_real"])
                batch_data["cluster_graph"]["offsets"].append(data["cluster_graph"]["offsets"])
                batch_data["cluster_graph"]["rvecs"].append(data["cluster_graph"]["rvecs"])
                batch_data["cluster_graph"]["batch"].append(torch.full((data["cluster_graph"]["pos"].shape[0], ), i))
            
                batch_data["underling_network"]["inter"].append(data["underling_network"]["inter"] + torch.tensor([[base_cluster_idx], [base_network_idx]]))
                batch_data["underling_network"]["edges"].append(data["underling_network"]["edges"] + base_network_idx)
                batch_data["underling_network"]["pos"].append(data["underling_network"]["pos"])
                batch_data["underling_network"]["offsets_real"].append(data["underling_network"]["offsets_real"])
                batch_data["underling_network"]["offsets"].append(data["underling_network"]["offsets"])
                batch_data["underling_network"]["rvecs"].append(data["underling_network"]["rvecs"])
                batch_data["underling_network"]["batch"].append(torch.full((data["underling_network"]["pos"].shape[0], ), i))
                
                base_atom_radius_idx += data["atom_radius_graph"]["pos"].shape[0]
                base_atom_bond_idx += data["atom_bond_graph"]["pos"].shape[0]
                base_cluster_idx += data["cluster_graph"]["pos"].shape[0]
                base_network_idx += data["underling_network"]["pos"].shape[0]

            for descriptor in self.descriptor_index:
                batch_data[descriptor].append(data[descriptor])
            batch_data["name"].append(data["name"])
        
        if "atom_radius_graph" in self.used_topos:
            batch_data["atom_radius_graph"]["numbers"] = torch.cat(batch_data["atom_radius_graph"]["numbers"], dim=0)
            batch_data["atom_radius_graph"]["edges"] = torch.cat(batch_data["atom_radius_graph"]["edges"], dim=1)
            batch_data["atom_radius_graph"]["pos"] = torch.cat(batch_data["atom_radius_graph"]["pos"], dim=0)
            batch_data["atom_radius_graph"]["offsets_real"] = torch.cat(batch_data["atom_radius_graph"]["offsets_real"], dim=0)
            batch_data["atom_radius_graph"]["offsets"] = torch.cat(batch_data["atom_radius_graph"]["offsets"], dim=0)
            batch_data["atom_radius_graph"]["batch"] = torch.cat(batch_data["atom_radius_graph"]["batch"], dim=0)
        if "atom_bond_graph" in self.used_topos:
            batch_data["atom_bond_graph"]["numbers"] = torch.cat(batch_data["atom_bond_graph"]["numbers"], dim=0)
            batch_data["atom_bond_graph"]["edges"] = torch.cat(batch_data["atom_bond_graph"]["edges"], dim=1)
            batch_data["atom_bond_graph"]["pos"] = torch.cat(batch_data["atom_bond_graph"]["pos"], dim=0)
            batch_data["atom_bond_graph"]["offsets_real"] = torch.cat(batch_data["atom_bond_graph"]["offsets_real"], dim=0)
            batch_data["atom_bond_graph"]["offsets"] = torch.cat(batch_data["atom_bond_graph"]["offsets"], dim=0)
            batch_data["atom_bond_graph"]["batch"] = torch.cat(batch_data["atom_bond_graph"]["batch"], dim=0)
        if "high_order" in self.used_topos:
            batch_data["cluster_graph"]["inter"] = torch.cat(batch_data["cluster_graph"]["inter"], dim=1)
            batch_data["cluster_graph"]["edges"] = torch.cat(batch_data["cluster_graph"]["edges"], dim=1)
            batch_data["cluster_graph"]["pos"] = torch.cat(batch_data["cluster_graph"]["pos"], dim=0)
            batch_data["cluster_graph"]["offsets_real"] = torch.cat(batch_data["cluster_graph"]["offsets_real"], dim=0)
            batch_data["cluster_graph"]["offsets"] = torch.cat(batch_data["cluster_graph"]["offsets"], dim=0)
            batch_data["cluster_graph"]["batch"] = torch.cat(batch_data["cluster_graph"]["batch"], dim=0)
        
            batch_data["underling_network"]["inter"] = torch.cat(batch_data["underling_network"]["inter"], dim=1)
            batch_data["underling_network"]["edges"] = torch.cat(batch_data["underling_network"]["edges"], dim=1)
            batch_data["underling_network"]["pos"] = torch.cat(batch_data["underling_network"]["pos"], dim=0)
            batch_data["underling_network"]["offsets_real"] = torch.cat(batch_data["underling_network"]["offsets_real"], dim=0)
            batch_data["underling_network"]["offsets"] = torch.cat(batch_data["underling_network"]["offsets"], dim=0)
            batch_data["underling_network"]["batch"] = torch.cat(batch_data["underling_network"]["batch"], dim=0)

        for descriptor in self.descriptor_index:
            batch_data[descriptor] = torch.cat(batch_data[descriptor], dim=0)
        return batch_data