# MOFTransformer version 2.0.0
import copy
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable
import numpy as np
import torch
import lightning.pytorch as lp

from crystalproject.module.predictor_module import PreModule
from crystalproject.data.map_data_module import MapDataModule
from crystalproject.config import get_config


@lru_cache
def get_model_and_datamodule(path, model_path):
    _, data_config = get_config(path)
    
    lp.seed_everything(123)
    model = PreModule.load_from_checkpoint(model_path)
    model.eval()
    model.to("cpu")

    data_config["dataloader"] = {
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
    }
    dm = MapDataModule(**data_config)
    dm.setup("test")
    data_iter = dm.test_dataloader()

    return model, data_iter


@lru_cache
def get_batch_from_cif_name(data_iter, cif_name):
    cif_name = Path(cif_name).stem
    iter_ = iter(data_iter)
    while True:
        try:
            batch = next(iter_)
        except StopIteration:
            raise ValueError(f"There are no {cif_name} in dataset")
        else:
            batch_id = batch["name"][0]
            if batch_id == cif_name:
                return batch


def get_heatmap(out, batch_idx, graph_len=300, skip_cls=True):
    """
    attention rollout  in "Quantifying Attention Flow in Transformers" paper.
    :param out: output of model.infer(batch)
    :param batch_idx: batch index
    :param graph_len: the length of grid embedding
    :param skip_cls: <bool> If True, class token is ignored.
    :return: <np.ndarray> heatmap graph, heatmap grid
    """
    attn_weights = torch.stack(
        out["attn_weights"]
    )  # [num_layers, B, num_heads, max_len, max_len]
    att_mat = attn_weights[:, batch_idx]  # [num_layers, num_heads, max_len, max_len]

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)  # [num_layers, max_len, max_len]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att

    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(
        -1
    )  # [num_layers, max_len, max_len]
    aug_att_mat = aug_att_mat.detach().numpy()  # prevent from memory leakage

    # Recursively multiply the weight matrices
    joint_attentions = np.zeros(aug_att_mat.shape)  # [num_layers, max_len, max_len]
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]  # [max_len, max_len]

    # Don't drop class token when normalizing
    if skip_cls:
        v_ = v[0][1:]  # skip cls token
        cost_graph = v_[:graph_len]  # / v_.max()
        cost_grid = v_[graph_len:]  # / v_.max()
        heatmap_graph = cost_graph
        heatmap_grid = cost_grid[1:-1].reshape(6, 6, 6)  # omit cls + volume tokens
    else:
        v_ = v[0]
        cost_graph = v_[: graph_len + 1]  # / v_.max()
        cost_grid = v_[graph_len + 1 :]  # / v_.max()
        heatmap_graph = cost_graph[1:]  # omit cls token
        heatmap_grid = cost_grid[1:-1].reshape(6, 6, 6)  # omit cls + volume tokens

    return heatmap_graph, heatmap_grid


def scaler(value, min_att, max_att):
    if isinstance(value, float):
        if value > max_att:
            value = max_att
        elif value < min_att:
            value = min_att
        return float((value - min_att) / (max_att - min_att))

    elif isinstance(value, np.ndarray):
        value = copy.deepcopy(value)
        value[value > max_att] = max_att
        value[value < min_att] = min_att
        return (value - min_att) / (max_att - min_att)
    elif isinstance(value, Iterable):
        return scaler(np.array(list(value), dtype="float"), min_att, max_att)
    else:
        raise TypeError(f"value must be float, list, or np.array, not {type(value)}")
