import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.utils.normalize import Normalizer


class PreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_loss()
        self.configure_criterion()
        self.configure_normalize()
        self.test_out_output = []

    def configure_model(self):
        conf_model = self.hparams["model"]
        model_cls = registry.get_model_class(conf_model["name"])
        self.model = model_cls(**conf_model["kwargs"])

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
                self.loss_function = F.mse_loss
            case "l1":
                self.loss_function = F.l1_loss
            case "bce":
                self.loss_function = F.binary_cross_entropy
            case _:
                self.print("Loss not found.")

    def configure_criterion(self):
        conf_criterion = self.hparams["criterion"]
        match conf_criterion["name"]:
            case "mae":
                self.criterion_function = F.l1_loss
            case _:
                self.print("Loss not found.")

    def configure_normalize(self):
        conf_normalize = self.hparams["normalize"]
        self.normalize = Normalizer(**conf_normalize)

    def forward(self, input):
        return self.model(*input)

    def training_step(self, batch, batch_idx):
        input, output = batch[0], batch[1]
        out = self(input)
        loss = self.loss_function(out, self.normalize.norm(output))
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=output.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        input, output = batch
        out = self(input)
        loss = self.loss_function(out, self.normalize.norm(output))
        criterion = self.criterion_function(self.normalize.denorm(out), output)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=output.shape[0])
        self.log('val_criterion', criterion, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=output.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        input, output = batch
        out = self(input)
        self.test_out_output.append(
            torch.cat([self.normalize.denorm(out), output], dim=1)
        )
        criterion = self.criterion_function(self.normalize.denorm(out), output)
        self.log('test_criterion', criterion, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=output.shape[0])
        return criterion

    def on_test_epoch_end(self):
        config = self.hparams["config"]
        out_output = torch.cat(self.test_out_output, dim=0).cpu()
        out_output = out_output.numpy()
        out = out_output[:, 0]
        output = out_output[:, 1]
        fig, ax = plt.subplots(figsize=(5, 5))
        Axis_line = np.linspace(*ax.get_xlim(), 2)
        ax.plot(Axis_line, Axis_line, transform=ax.transAxes,
                linestyle='--', linewidth=2, color='black', label=config["name"])
        ax.scatter(out, output, color='red')
        ax.legend()
        plt.savefig(os.path.join(config["root_dir"], config["name"]+'.png'),
                    bbox_inches='tight')


if __name__ == "__main__":
    conf = {
        "model": {
            "name": "cgcnn",
            "kwargs": {
                "orig_atom_fea_len": 92,
                "nbr_fea_len": 41
            }
        },
        "optimizers": {
            "name": "Adam",
            "kwargs": {
                "lr": 0.01,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "name": "StepLR",
            "kwargs": {
                "step_size": 0.1,
                "gamma": 0.1
            }
        },
        "loss": {
            "name": "mse"
        },
        "criterion": {
            "name": "mae"
        },
        "normalize": {
            "mean": 0,
            "std": 1
        }
    }
    m = PreModule(**conf)
