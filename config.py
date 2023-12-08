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
                    "edge_embedding":{
                        "dmin": 1.0,
                        "dmax": 2.0,
                        "step": 0.1,
                    },
                    "num_layers": 3
                }
            },
            "cluster_hidden_channels": 256,
            "cluster_graph":{
                "name": "GCN",
                "kwargs":{
                    "num_layers": 2,
                }
            },
            "underling_network":{
                "name": "cgcnn",
                "kwargs":{
                    "edges_embedding":{
                        "dmin": 10.0,
                        "dmax": 20.0,
                        "step": 1.0,
                    },
                    "num_layers": 3
                }
            }
        }
    },
    "head":{
        "name": "mlphead",
        "kwargs":{
            "in_channels": 256,
            "hidden_channels": 256,
            "out_channels": 1,
            "n_h": 3,
        }
    },
    "optimizers":{
        "name": "Adam",
        "kwargs":{
            
        },
    },
    "scheduler":{
        "name": "StepLR",
        "kwargs":{
            "step_size": 500
        },
    },
    "loss":{
        "name": "mse",
    },
    "normalize":{
        "mean": 0,
        "std": 1,
    }
},
data_config = {
    "dataset":{
        "name": "CrystalTopoDataset",
        "kwargs":{
            "root_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process",
            "target_index": ["absolute methane uptake high P [v STP/v]"]
        }
    }
},
trainer_config = {
    "max_epochs": 500,
    "min_epochs": 100
},
config = {
    "root_dir": "",
    "target": "CH4 High P"
},


# 回调函数
early_stop = EarlyStopping(
    monitor="val_criterion",
    mode="min",
)

model_checkpoint = ModelCheckpoint(
    dirpath="checkpoints",
    filename="model-{epoch:02d}-{val_criterion:.2f}",
    save_top_k=3,
    monitor="val_criterion",
    mode="min"
)


# 主流程
module = PreModule(**module_config)
data_module = MapDataModule(**data_config)
trainer = lp.Trainer(**trainer_config, callbacks=[early_stop, model_checkpoint], devices=[7])
trainer.fit(module, data_module)
save_path = trainer.default_root_dir
config["root_dir"] = save_path
trainer.test(module, data_module, config)

