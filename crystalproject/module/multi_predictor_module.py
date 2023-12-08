import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as lp
from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.utils.normalize import Normalizer


class MultiPreModule(lp.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_loss()
        self.configure_criterion()
        self.test_out_output = []

    def configure_model(self):
        conf_backbone = self.hparams["backbone"]
        model_cls = registry.get_model_class(conf_backbone["name"])
        self.backbone = model_cls(**conf_backbone["kwargs"])
        conf_head = self.hparams["head"]
        head_cls = registry.get_head_class(conf_head["name"])
        self.heads = nn.ModuleList(
            [
                head_cls(**conf_head["kwargs"])
                for _ in range(conf_head["number"])
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
        conf_loss = self.hparams["loss"]
        loss_cls = registry.get_loss_class(conf_loss["name"])
        self.loss = loss_cls(**conf_loss["kwargs"])

    def configure_criterion(self):
        conf_criterion = self.hparams["criterion"]
        match conf_criterion["name"]:
            case "mae":
                self.criterion = F.l1_loss
            case _:
                self.print("Criterion not found.")

    def forward(self, input):
        out = self.backbone(*input)
        out = torch.cat([head(out) for head in self.heads], dim=1)
        return out

    def training_step(self, batch, batch_idx):
        input, output = batch[0], batch[1]
        out = self(input)
        loss = self.loss(out, output)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=output.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        input, output = batch
        out = self(input)
        for i in range(out.size()[1]):
            criterion = self.criterion(out[:, i], output[:, i])
            self.log(self.hparams["config"]["name"][i]+"_val_criterion", criterion, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=output.shape[0]) 


    def test_step(self, batch, batch_idx):
        input, output = batch
        out = self(input)
        self.test_out_output.append(
            torch.cat([out, output], dim=1)
        )

    def on_test_epoch_end(self, config):
        out_output = torch.cat(self.test_out_output, dim=0)
        number = self.hparams["head"]["number"]
        out = out_output[:, :number]
        output = out_output[:, number:]
        for i in range(number):
            criterion = self.criterion(out[:, i], output[:, i])
            _, ax = plt.subplots(figsize=(5, 5))
            plt.xlabel(config["target"][i]+" predict value")
            plt.ylabel(config["target"][i]+" true value")
            plt.text(5, 0, f'{self.hparams["criterion"]["name"]} is {criterion}')
            Axis_line = np.linspace(*ax.get_xlim(), 2)
            ax.plot(Axis_line, Axis_line, transform=ax.transAxes,
                    linestyle='--', linewidth=2, color='black', label=config["target"][i])
            ax.scatter(out[:, i].cpu(), output[:, i].cpu(), color='red')
            ax.legend()
            plt.savefig(os.path.join(config["root_dir"], config["name"][i]+'.png'),
                        bbox_inches='tight')

