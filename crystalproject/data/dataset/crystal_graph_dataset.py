import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from crystalproject.utils.registry import registry


@registry.register_dataset("CrystalGraphDataset")
class CraystalGraphDataset(Dataset):
    def __init__(self, root_dir, stage="predict"):
        match stage:
            case "train" | "val" | "test":
                self.root_dir = os.path.join(root_dir, stage)
            case _:
                self.root_dir = root_dir
        with open(os.path.join(self.root_dir, "id_prop.json"), "r") as f:
            self.id_prop_data = json.load(f)
            self.ids = list(self.id_prop_data.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        data = np.load(os.path.join(self.root_dir, id+".npz"))
        atom_fea = data["atom_fea"]
        nbr_fea = data["nbr_fea"]
        nbr_fea_idx = data["nbr_fea_idx"]
        atom_fea = torch.tensor(atom_fea, dtype=torch.int32)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.int64)
        target = torch.tensor([self.id_prop_data[id]], dtype=torch.float32)
        return (atom_fea, nbr_fea, nbr_fea_idx), target

    @staticmethod
    def collate(dataset_list):
        (
            batch_atom_fea,
            batch_nbr_fea,
            batch_nbr_fea_idx,
            crystal_atom_idx,
            batch_target,
        ) = ([], [], [], [], [])
        base_idx = 0
        for _, ((atom_fea, nbr_fea, nbr_fea_idx), target) in enumerate(
            dataset_list
        ):
            n_i = atom_fea.shape[0]
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            crystal_atom_idx.append(n_i)
            batch_target.append(target)
            base_idx += n_i
        return (
            (
                torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=1),
                np.array(crystal_atom_idx),
            ),
            torch.stack(batch_target, dim=0),
        )
