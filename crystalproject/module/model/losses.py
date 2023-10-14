from torch.nn import functional as F

from crystalproject.utils.registry import registry


@registry.register_loss("multiLoss")
class MulitLoss():
    def __init__(self, name, theta):
        match name:
            case "mse":
                self.function = F.mse_loss
            case "l1":
                self.function = F.l1_loss
            case "bce":
                self.function = F.binary_cross_entropy
            case _:
                self.print("Loss not found")
        self.theta = theta
    
    def __call__(self, x, y):
        loss = self.function(x, y)
        loss += self.theta * (self.function(x[:, 0] - x[:, 1], x[:, 2]))
        return loss
