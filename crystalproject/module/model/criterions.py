from torch.nn import functional as F

from crystalproject.utils.registry import registry


@registry.register_criterion("multiCriterion")
class MulitCriterion():
    def __init__(self, name):
        match name:
            case "mae":
                self.function = F.l1_loss
            case _:
                self.print("Criterion not found.")
    
    def __call__(self, x, y):
        return self.function(x, y)