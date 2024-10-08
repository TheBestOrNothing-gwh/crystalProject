import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanSquaredError

from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.utils import *


class PreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config_model()
        self.configure_loss()
        self.configure_metrics()

    def config_model(self):
        conf_backbone = self.hparams["backbone"]
        model_cls = registry.get_model_class(conf_backbone["name"])
        self.backbone = model_cls(**conf_backbone["kwargs"])
        conf_predictor = self.hparams["predictor"]
        self.heads = nn.ModuleList(
            [
                registry.get_head_class(conf_head["name"])(**conf_head["kwargs"])
                for conf_head in conf_predictor["heads"]
            ]
        )
        conf_normalizers = self.hparams["normalizers"]
        self.normalizers = nn.ModuleDict(
            {
                key: registry.get_model_class(value["name"])(**value["kwargs"])
                for key, value in conf_normalizers.items()
            }
                
        )
    
    def configure_optimizers(self):
        conf_optimizer = self.hparams["optimizers"]
        conf_scheduler = self.hparams["scheduler"]
        match conf_optimizer["name"]:
            case "Adam":
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    **conf_optimizer["kwargs"]
                )
            case "SGD":
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    **conf_optimizer["kwargs"]
                )
            case _:
                self.print("Optimizer not found.")
        match conf_scheduler["name"]:
            case "StepLR":
                scheduler = lrs.StepLR(optimizer, **conf_scheduler["kwargs"])
            case "CosineAnnealingLR":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer, **conf_scheduler["kwargs"])
            case "OneCycleLR":
                scheduler = lrs.OneCycleLR(
                    optimizer, **conf_scheduler["kwargs"]
                )
        return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss = F.mse_loss
        self.weight = self.hparams["predictor"]["targets"]
            
    def configure_metrics(self):
        self.mses = nn.ModuleDict({target: MeanSquaredError() for target in self.hparams["predictor"]["targets"].keys()})
        self.maes = nn.ModuleDict({target: MeanAbsoluteError() for target in self.hparams["predictor"]["targets"].keys()})
        self.r2s = nn.ModuleDict({target: R2Score() for target in self.hparams["predictor"]["targets"].keys()})
        
    def forward(self, batch):
        self.backbone(batch)
        batch["output"] = {}
        for head in self.heads:
            head(batch)
    
    def training_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        total_loss = sum([self.weight[target] * self.loss(out[target], self.normalizers[target].norm(batch[target])) for target in self.hparams["predictor"]["targets"].keys()])
        self.log('train_loss', total_loss, prog_bar=False, batch_size=batch["batch_size"])
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        for target in self.hparams["predictor"]["targets"].keys():
            self.mses[target].update(self.normalizers[target].denorm(out[target]), batch[target])
            self.log(f'val_mse_{target}', self.mses[target], prog_bar=False, batch_size=batch["batch_size"])
            self.maes[target].update(self.normalizers[target].denorm(out[target]), batch[target])
            self.log(f'val_mae_{target}', self.maes[target], prog_bar=False, batch_size=batch["batch_size"])
    
    def test_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        for target in self.hparams["predictor"]["targets"].keys():
            self.maes[target].update(self.normalizers[target].denorm(out[target]), batch[target])
            self.log(f'test_mae_{target}', self.maes[target], prog_bar=False, batch_size=batch["batch_size"])
            self.r2s[target].update(self.normalizers[target].denorm(out[target]), batch[target])
            self.log(f'test_r2_{target}', self.r2s[target], prog_bar=False, batch_size=batch["batch_size"])
