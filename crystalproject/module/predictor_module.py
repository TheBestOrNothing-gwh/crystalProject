import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from torchmetrics.regression import MeanAbsoluteError, R2Score

from crystalproject.utils.registry import registry
from crystalproject.module.model import *


class PreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_loss()
        self.configure_metrics()

    def configure_model(self):
        conf_backbone = self.hparams["backbone"]
        model_cls = registry.get_model_class(conf_backbone["name"])
        self.backbone = model_cls(**conf_backbone["kwargs"])
        conf_heads = self.hparams["heads"]
        self.heads = nn.ModuleList(
            [
                registry.get_head_class(conf_head["name"])(**conf_head["wargs"])
                for conf_head in conf_heads
            ]
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
        return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss = F.mse_loss
        self.weight = self.hparams["heads"]["targets"]
            
    def configure_metrics(self):
        self.maes = dict.fromkeys(self.hparams["heads"]["targets"].keys(), MeanAbsoluteError())
        self.r2s = dict.fromkeys(self.hparams["heads"]["targets"].keys(), R2Score())
        
    def forward(self, batch):
        self.backbone(batch)
        batch["output"] = {}
        for head in self.heads:
            head(batch)
    
    def training_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        total_loss = sum([self.weight[target] * self.loss(out[target], batch[target]) for target in self.hparams["heads"]["target"]])
        self.log('train_loss', total_loss, prog_bar=True, batch_size=batch["batch"]["batch_size"])
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        for target in self.hparams["heads"]["target"]:
            self.maes[target].update(out[target], batch[target])
            self.log(f'val_mae_{target}', self.maes[target], prog_bar=True, batch_size=batch["batch"]["batch_size"])
            self.r2s[target].update(out[target], batch[target])
            self.log(f'val_r2_{target}', self.r2s[target], prog_bar=True, batch_size=batch["batch"]["batch_size"])
    
    def test_step(self, batch, batch_idx):
        self(batch)
        out = batch["output"]
        for target in self.hparams["heads"]["target"]:
            self.maes[target].update(out[target], batch[target])
            self.log(f'test_mae_{target}', self.maes[target], prog_bar=True, batch_size=batch["batch"]["batch_size"])
            self.r2s[target].update(out[target], batch[target])
            self.log(f'test_r2_{target}', self.r2s[target], prog_bar=True, batch_size=batch["batch"]["batch_size"])