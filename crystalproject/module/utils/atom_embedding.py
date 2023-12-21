import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import numpy as np
from math import sqrt


class AtomEmbedding(nn.Module):
    def __init__(self, config_path, hidden_channels):
        super(AtomEmbedding, self).__init__()
        with open(config_path) as f:
            elem_embedding = json.load(f)
            elem_embedding = {int(key): torch.tensor(np.array([value]), dtype=torch.float32) for key,
                              value in elem_embedding.items()}
        embedding = []
        for i in range(len(list(elem_embedding.keys()))):
            embedding.append(elem_embedding[i + 1])
        self.embedding = nn.Parameter(
            torch.cat(embedding, dim=0), requires_grad=False)
        self.lin = nn.Linear(self.embedding.size(1), hidden_channels, bias=False)

    def forward(self, x):
        return self.lin(F.embedding(x, self.embedding))


class AtomEmbeddingNoPriori(nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEmbeddingNoPriori, self).__init__()
        self.node_embedding = nn.Embedding(95, hidden_channels)

    def reset_parameters(self):
        self.node_embedding.weight.data.uniform_(-sqrt(3), sqrt(3))
    
    def forward(self, x):
        return self.node_embedding(x)

