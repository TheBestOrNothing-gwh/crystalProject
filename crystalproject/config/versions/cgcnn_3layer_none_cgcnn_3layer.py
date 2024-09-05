from molmod.units import angstrom


# 配置文件
module_config = {
    "backbone":{
        "name": "toponet",
        "kwargs":{
            "atom_embedding":{
                "config_path": "/home/gwh/project/crystalProject/models/crystalProject/crystalproject/assets/atom_init.json"
            },
            "atom_hidden_channels": 128,
            "atom_radius_graph":{
                "name": "cgcnn",
                "kwargs":{
                    "num_layers": 3,
                    "edge_embedding":{
                        "dmin": 0.0,
                        "dmax": 8.0 * angstrom,
                        "step": 0.2 * angstrom
                    }
                }
            },
            "cluster_hidden_channels": 128,
            "cluster_graph":{
                "name": "cgcnn",
                "kwargs":{
                    "num_layers": 3,
                    "edge_embedding":{
                        "dmin": 0.0,
                        "dmax": 80.0 * angstrom,
                        "step": 2.0 * angstrom
                    }
                }
            },
            "underling_network":{
                "name": "cgcnn",
                "kwargs":{
                    "num_layers": 3,
                    "edge_embedding":{
                        "dmin": 0.0,
                        "dmax": 80.0 * angstrom,
                        "step": 2.0 * angstrom
                    }
                }
            }
        }
    },
    "predictor":{
        "targets": {"absolute methane uptake high P [v STP/v]": 1.0},
        "heads":[
            {
                "name": "mlphead",
                "kwargs":{
                    "in_channels": 256,
                    "out_channels": 1,
                    "targets": ["absolute methane uptake high P [v STP/v]"],
                    "descriptors": ["atom_radius_graph_readout", "underling_network_readout"]
                }
            },
        ]
    },
    "optimizers":{
        "name": "Adam",
        "kwargs":{
            "lr":5e-4,
            # "weight_decay": 1e-5
        },
    },
    "scheduler":{
        "name": "OneCycleLR",
        "kwargs":{
            "max_lr": 5e-4,
            "total_steps": 1000,
            # "step_size": 500,
            # "gamma": 0.1
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
            "input_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/input_dir",
            "split_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/split_dir/random1",
            "descriptor_index": ["absolute methane uptake high P [v STP/v]"],
            "used_topos": ["atom_radius_graph", "atom_bond_graph", "high_order"]
        }
    },
    "dataloader":{
        "batch_size": 64,
        "num_workers": 16,
        "pin_memory": True,
    }
}