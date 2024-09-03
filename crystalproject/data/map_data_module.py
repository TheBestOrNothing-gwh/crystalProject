import lightning.pytorch as lp
from torch.utils.data import DataLoader


from crystalproject.utils.registry import registry
from crystalproject.data.dataset import *
from crystalproject.data.utils.dataloader import MultiEpochsDataLoader, CudaDataLoader


class MapDataModule(lp.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        conf_dataset = self.hparams["dataset"]
        dataset_cls = registry.get_dataset_class(conf_dataset["name"])
        match stage:
            case "fit":
                self.train_dataset = dataset_cls(
                    stage="train", **conf_dataset["kwargs"])
                self.val_dataset = dataset_cls(
                    stage="val", **conf_dataset["kwargs"])
            case "test":
                self.test_dataset = dataset_cls(
                    stage="test", **conf_dataset["kwargs"])
            case "predict":
                self.predict_dataset = dataset_cls(
                    stage="predict", **conf_dataset["kwargs"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, collate_fn=self.train_dataset.collate, **self.hparams["dataloader"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, collate_fn=self.val_dataset.collate, **self.hparams["dataloader"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, collate_fn=self.test_dataset.collate, **self.hparams["dataloader"])

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, shuffle=False, collate_fn=self.predict_dataset.collate, **self.hparams["dataloader"])


if __name__ == "__main__":
    conf = {
        "dataset": {
            "name": "CIFData",
            "kwargs": {
                "root_dir": "/home/gwh/project/crystalProject/DATA/qmof_database/debug_structures",
            }
        },
        "split": [0.8, 0.1, 0.1],
        "dataloader": {
            "batch_size": 2,
            "num_workers": 8,
            "pin_memory": True,
        }
    }
    m = MapDataModule(**conf)
    m.setup("predict")
    dataloader = m.predict_dataloader()
    for _, (input, output) in enumerate(dataloader):
        print(input[0].shape)
        print(input[1].shape)
        print(input[2].shape)
        exit()
