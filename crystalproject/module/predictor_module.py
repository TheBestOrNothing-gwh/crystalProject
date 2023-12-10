from matplotlib.figure import Figure
import os
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from sklearn.metrics import mean_absolute_error, r2_score

from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.utils.normalize import Normalizer
from crystalproject.visualize.drawer import draw_compare


class PreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_loss()
        self.configure_criterion()
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
        conf_loss = self.hparams["loss"]
        match conf_loss["name"]:
            case "mse":
                self.loss = F.mse_loss
            case "l1":
                self.loss = F.l1_loss
            case "bce":
                self.loss = F.binary_cross_entropy
            case _:
                self.print("Loss not found.")
    
    def configure_criterion(self):
        conf_criterion = self.hparams["criterion"]
        match conf_criterion["name"]:
            case "mae":
                self.criterion = F.l1_loss
            case _:
                self.print("Loss not found.")

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
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=batch["target"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        criterion = self.criterion(self.normalize.denorm(out), batch["target"])
        self.log('val_criterion', criterion, on_step=False,
            on_epoch=True, prog_bar=True, batch_size=batch["target"].shape[0])

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.test_value.append(batch["target"])
        self.test_pre.append(self.normalize.denorm(out))
    
    def set_config(self, config):
        self.config = config

    def on_test_epoch_end(self):
        config = self.config
        test_value = torch.cat(self.test_value, dim=0).cpu()
        test_pre = torch.cat(self.test_pre, dim=0).cpu()
        addition = "\n".join(
            [
                f"MAE = {round(mean_absolute_error(test_value, test_pre), 2)}",
                f"R2 = {round(r2_score(test_value, test_pre), 2)}",
            ]
        )
        fig = Figure(
            figsize=(8, 8),
            dpi=300
        )
        ax = fig.add_subplot()
        draw_compare(
            fig=fig,
            ax=ax,
            x=test_value,
            y=test_pre,
            x_label="Mol.Sim",
            y_label="ML",
            addition=addition,
            title=config["target"]
        )
        # 找到当前文件运行的位置
        fig.savefig(os.path.join(config["root_dir"], "对比密度图.png"), bbox_inches="tight")
