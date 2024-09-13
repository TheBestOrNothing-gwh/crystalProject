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
                "name": "schnet",
                "kwargs":{
                    "num_layers": 3,
                    "cutoff": 5.0 * angstrom
                }
            },
            "atom_bond_graph":{},
            "cluster_hidden_channels": 128,
            "cluster_graph":{},
            "linker_graph":{},
            "underling_network":{}
        }
    },
    "predictor":{
        "targets": {
            "absolute methane uptake high P [v STP/v]": 0.01, 
            "absolute methane uptake low P [v STP/v]": 1.,
            "CO2 Qst [kJ/mol]": 1.,
            "CO2 kH [mol/kg/Pa] log": 1.
        },
        "heads":[
            {
                "name": "mlphead",
                "kwargs":{
                    "in_channels": 128,
                    "out_channels": 4,
                    "targets": [
                        "absolute methane uptake high P [v STP/v]", 
                        "absolute methane uptake low P [v STP/v]",
                        "CO2 Qst [kJ/mol]",
                        "CO2 kH [mol/kg/Pa] log"
                    ],
                    "descriptors": [
                        "atom_radius_graph_readout"
                    ]
                }
            },
        ]
    },
    "optimizers":{
        "name": "Adam",
        "kwargs":{
            "lr":5e-4
        },
    },
    "scheduler":{
        "name": "OneCycleLR",
        "kwargs":{
            "max_lr": 5e-4,
            "total_steps": 1000
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
            "input_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/input_dir/5a",
            "split_dir": "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/split_dir/random1",
            "descriptor_index": [
                "absolute methane uptake high P [v STP/v]", 
                "absolute methane uptake low P [v STP/v]",
                "CO2 Qst [kJ/mol]",
                "CO2 kH [mol/kg/Pa] log"
            ],
            "used_topos": [
                "atom_radius_graph", 
                "atom_bond_graph", 
                "high_order"
            ]
        }
    },
    "dataloader":{
        "batch_size": 64,
        "num_workers": 16,
        "pin_memory": True,
    }
}
trainer_config = {
    "max_epochs": 1000,
    "min_epochs": 100,
    "default_root_dir": "log/",
    "accumulate_grad_batches":1
}