from matplotlib.figure import Figure
import os
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from torchmetrics.regression import MeanAbsoluteError, R2Score

from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.utils.normalize import Normalizer


class PreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_loss()
        self.configure_metrics()
        self.configure_normalize()
        self.test_value = []
        self.test_pre = []

    def configure_model(self):
        conf_backbone = self.hparams["backbone"]
        model_cls = registry.get_model_class(conf_backbone["name"])
        self.backbone = model_cls(**conf_backbone["kwargs"])
        conf_head = self.hparams["head"]
        head_cls = registry.get_head_class(conf_head["name"])
        self.head = head_cls(**conf_head["kwargs"])

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

    def configure_metrics(self):
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        
    def configure_normalize(self):
        conf_normalize = self.hparams["normalize"]
        self.normalize = Normalizer(**conf_normalize)

    def forward(self, input):
        out = self.backbone(input)
        out = self.head(out)
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, self.normalize.norm(batch["target"]))
        self.log('train_loss', loss, prog_bar=True, batch_size=batch["target"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.mae.update(self.normalize.denorm(out), batch["target"])
        self.log('val_mae', self.mae, prog_bar=True, batch_size=batch["target"].shape[0])

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.mae.update(self.normalize(out), batch["target"])
        self.r2.update(self.normalize(out), batch["target"])
        self.log("test_mae", self.mae, batch_size=batch["target"].shape[0])
        self.log("test_r2", self.r2, batch_size=batch["target"].shape[0])