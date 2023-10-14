import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import numpy as np


class NodeEmbedding(nn.Module):
    def __init__(self, config_path):
        super(NodeEmbedding, self).__init__()
        with open(config_path) as f:
            elem_embedding = json.load(f)
            elem_embedding = {int(key): torch.tensor(np.array([value]), dtype=torch.float32) for key,
                              value in elem_embedding.items()}
        embedding = []
        for i in range(len(list(elem_embedding.keys()))):
            embedding.append(elem_embedding[i + 1])
        self.embedding = nn.Parameter(
            torch.cat(embedding, dim=0), requires_grad=False)
        self.dim = self.embedding.size(1)

    def get_dim(self):
        return self.dim

    def forward(self, x):
        return F.embedding(x, self.embedding)
