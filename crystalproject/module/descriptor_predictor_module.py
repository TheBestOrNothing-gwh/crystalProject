from crystalproject.utils.registry import registry
from crystalproject.module.model import *
from crystalproject.module.predictor_module import PreModule

class DescPreModule(PreModule):
    def __init__(self, **kwargs):
        super().__init__()
    
    def configure_model(self):
        conf_head = self.hparams["head"]
        head_cls = registry.get_head_class(conf_head["name"])
        self.head = head_cls(**conf_head["kwargs"])
    
    def forward(self, input):
        out = self.head(input)
        return out
    