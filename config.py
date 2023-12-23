import lightning.pytorch as lp
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from crystalproject.module.predictor_module import PreModule
from crystalproject.data.map_data_module import MapDataModule


# 配置文件
module_config = {
    "backbone":{
        "name": "toponet",
        "kwargs":{
            "atom_embedding":{
                "config_path": "/home/gwh/project/crystalProject/models/crystalProject/crystalproject/assets/atom_init.json"
            },
            "atom_hidden_channels": 128,
            "atom_graph":{
                "name": "cgcnn",
                "kwargs":{
                    "num_layers": 3,
                    "edge_embedding": {"dmin": 0.0, "dmax": 5.0, "step": 0.2},
                }
            },
            "cluster_hidden_channels": 128,
            "cluster_graph":{
                "name": "gcn",
                "kwargs":{
                    "num_layers": 2,
                }
            },
            "underling_network":{
                "name": "cgcnn",
                "kwargs":{
                    "num_layers": 3,
                    "edge_embedding": {"dmin": 0.0, "dmax": 15.0, "step": 0.2},
                }
            }
        }
    },
    "predictor":{
        "targets": {"absolute methane uptake high P [v STP/v]":0.5, "absolute methane uptake low P [v STP/v]":0.5},
        "heads":[
            {
                "name": "regression",
                "kwargs":{
                    "in_channels": 137,
                    "out_channels": 2,
                    "targets": ["absolute methane uptake high P [v STP/v]", "absolute methane uptake low P [v STP/v]"],
                    "descriptors": ["atom_graph_embedding", "vol", "rho", "di", "df", "dif", "asa", "av", "nasa", "nav"]
                }
            },
        ]
    },
    "optimizers":{
        "name": "Adam",
        "kwargs":{
            "lr":5e-4,
            "weight_decay": 0.1
        },
    },
    "scheduler":{
        "name": "StepLR",
        "kwargs":{
            "step_size": 500,
        },
    },
    "loss":{
        "name": "mse",
    },
    "criterion":{
        "name": "mae",
    }
}
data_config = {
    "dataset":{
        "name": "CrystalTopoDataset",
        "kwargs":{
            "root_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process",
            "descriptor_index": ["absolute methane uptake high P [v STP/v]", "absolute methane uptake low P [v STP/v]", "vol", "rho", "di", "df", "dif", "asa", "av", "nasa", "nav"],
        }
    },
    "dataloader":{
        "batch_size": 16,
        "num_workers": 1,
        "pin_memory": True,
    }
}
trainer_config = {
    "max_epochs": 500,
    "min_epochs": 100
}


# 回调函数
early_stop = EarlyStopping(
    monitor="val_mae",
    mode="min",
)

model_checkpoint = ModelCheckpoint(
    filename="model-{epoch:02d}-{val_criterion:.2f}",
    save_top_k=3,
    monitor="val_mae_absolute methane uptake high P [v STP/v]",
    mode="min"
)


# 主流程
module = PreModule(**module_config)
data_module = MapDataModule(**data_config)
trainer = lp.Trainer(**trainer_config, devices=[2])
trainer.fit(module, data_module)
trainer.test(module, data_module)

