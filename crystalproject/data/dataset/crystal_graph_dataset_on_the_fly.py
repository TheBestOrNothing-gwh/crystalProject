import os
import json
import torch
from crystalproject.utils.registry import registry
from crystalproject.data.dataset.crystal_graph_dataset import CraystalGraphDataset
from crystalproject.data.prepare.process.crystal_graph import create_crystal_graph


@registry.register_dataset("CrystalGraphDatasetOnTheFly")
class CraystalGraphDatasetOnTheFly(CraystalGraphDataset):
    def __init__(self, root_dir, stage="predict", radius=8, max_nbr_num=12):
        super(CraystalGraphDatasetOnTheFly, self).__init__(root_dir, stage)
        self.radius = radius
        self.max_nbr_num = max_nbr_num

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        data = create_crystal_graph(
            os.path.join(self.root_dir, id+".cif"),
            self.radius,
            self.max_nbr_num
        )
        atom_fea = data["atom_fea"]
        nbr_fea = data["nbr_fea"]
        nbr_fea_idx = data["nbr_fea_idx"]
        atom_fea = torch.tensor(atom_fea, dtype=torch.int32)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.int64)
        target = torch.tensor([self.id_prop_data[id]], dtype=torch.float32)
        return (atom_fea, nbr_fea, nbr_fea_idx), target
