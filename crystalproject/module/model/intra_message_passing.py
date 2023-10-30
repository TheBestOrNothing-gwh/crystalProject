import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from crystalproject.utils.registry import registry


# intra message passing
@registry.register_model("intra_message_passing")
class IntraMessagePassing(MessagePassing):
    """
    实现同一层cells之间的消息传递
    """
    def __init__(self, ):
        TODO