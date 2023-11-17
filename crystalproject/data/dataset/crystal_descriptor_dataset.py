import torch
import os
import pandas as pd
import json
from torch.utils.data import Dataset

from crystalproject.utils.registry import registry


@registry.register_dataset("CrystalDescDataset")
class CrystalDescDataset(Dataset):
    def __init__(self, root_dir, stage="predict", descripter_index=[], target_index=[]):
        self.root_dir = root_dir
        match stage:
            case "train" | "val" | "test":
                self.id_prop = "id_prop_" + stage + ".json"
            case _:
                self.id_prop = "id_prop_all.json"
        self.descriptor_index = descripter_index
        self.target_index = target_index
        with open(os.path.join(self.root_dir, self.id_prop)) as f:
            datas = json.load(f)
        self.datas = pd.json_normalize(datas)
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        value = self.datas.loc[index]
        data = {}
        data["descriptor"] = torch.tensor([[value[i] for i in self.descriptor_index]], dtype=torch.float32)
        data["target"] = torch.tensor([[value[i] for i in self.target_index]], dtype=torch.float32)

        return data
    
    @staticmethod
    def collate(dataset_list):
        batch_data = {
            "descriptor": [],
            "target": []
        }
        
        for _, data in enumerate(dataset_list):
            batch_data["descriptor"].append(data["descriptor"])
            batch_data["target"].append(data["target"])
        
        batch_data["descriptor"] = torch.cat(batch_data["descriptor"], dim=0)
        batch_data["target"] = torch.cat(batch_data["target"], dim=0)

        return batch_data