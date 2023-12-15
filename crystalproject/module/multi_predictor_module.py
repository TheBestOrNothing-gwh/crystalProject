import os
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.regression import MeanAbsoluteError, R2Score

from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.predictor_module import PreModule
from crystalproject.visualize.drawer import draw_compare


class MultiPreModule(PreModule):
    def __init__(self, **kwargs):
        super().__init__()

    def configure_model(self):
        conf_backbone = self.hparams["backbone"]
        model_cls = registry.get_model_class(conf_backbone["name"])
        self.backbone = model_cls(**conf_backbone["kwargs"])
        conf_head = self.hparams["head"]
        head_cls = registry.get_head_class(conf_head["name"])
        self.heads = nn.ModuleDict(
            dict.fromkeys(
                conf_head["target"],
                head_cls(**conf_head["kwargs"])
            )
        )

    def configure_loss(self):
        self.loss = F.mse_loss
        
    
    def configure_metrics(self):
        self.maes = dict.fromkeys(self.hparams["head"]["target"], MeanAbsoluteError())
        self.r2s = dict.fromkeys(self.hparams["head"]["target"], R2Score())
        
    def forward(self, input):
        out = self.backbone(*input)
        out = {}
        for target, head in self.heads.items():
            out[target] = head(out)
        return out
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        for target in self.hparams["head"]["target"]:
            loss = self.loss(out[target], batch[target])
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        for target in self.hparams["head"]["target"]:
            self.maes[target].update(out[target], batch[target])
            self.log(f'val_{target}', self.maes[target], prog_bar=True, batch_size=batch[target].shape[0])

    def on_test_epoch_end(self):
        config = self.config
        test_value = torch.cat(self.test_value, dim=0).cpu()
        test_pre = torch.cat(self.test_pre, dim=0).cpu()
        num = len(config["target"])
        fig = Figure(
            figsize=(8, 8*num),
            dpi=300
        )
        for i in range(num):
            addition = "\n".join(
                [
                    f"MAE = {round(mean_absolute_error(test_value[:, i], test_pre[:, i]), 2)}",
                    f"R2 = {round(r2_score(test_value[:, i], test_pre[:, i]), 2)}",
                ]
            )
            ax = fig.add_subplot(1, num, i+1)
            draw_compare(
                fig=fig,
                ax=ax,
                x=test_value[:, i],
                y=test_pre[:, i],
                x_label="Mol.Sim",
                y_label="ML",
                addition=addition,
                title=config["target"][i]
            )
        fig.savefig(os.path.join(config["root_dir"], "对比密度图.png"), bbox_inches="tight")

