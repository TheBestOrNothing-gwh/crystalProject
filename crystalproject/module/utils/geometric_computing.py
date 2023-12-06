# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def crystal_to_dat(pos, edges, offsets, offsets_real):
    # Calculate distances. # number of edges
    dist = (pos[edges[1]] - pos[edges[0]] + offsets_real).norm(dim=-1)

    # Calculate angle
    # get triplets
    triplets = torch.nonzero(torch.eq(edges[1][:, None], edges[0])).T
    print(triplets)
    mask = (
        torch.eq(edges[:, triplets[0]][0], edges[:, triplets[1]][1]) & 
        torch.eq(offsets[triplets[0]], -offsets[triplets[1]]).all(dim=1)
    )
    print(mask)
    triplets = triplets[:, ~mask]
    # compute angle
    p_jk = -(pos[edges[:, triplets[0]][1]] - pos[edges[:, triplets[0]][0]] + offsets_real[triplets[0]])
    p_ji = pos[edges[:, triplets[1]][1]] - pos[edges[:, triplets[1]][0]] + offsets_real[triplets[1]]
    print(p_jk)
    print(p_ji)
    a = (p_jk * p_ji).sum(dim=-1)
    print(a)
    b = torch.cross(p_jk, p_ji).norm(dim=-1)
    angle = torch.atan2(b, a)

    # Calculate torsion

    # torsion = torch.nonzero(torch.eq(triplets[1][:, None], triplets[0])).T
    # mask = (
    #     torch.eq(triplets[:, torsion[0]][0], triplets[:, torsion[1]][1]) &
    #     torch.eq()
    # )
    return dist, edges, angle, triplets


if __name__ == "__main__":
    pos = torch.tensor(
        [
            [0.2, 0.2, 0.],
            [0.8, 0.2, 0.],
            [0.8, 0.8, 0.],
            [0.2, 0.8, 0.],
        ]
    )
    edges = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 3, 0, 2, 1, 3, 0, 2]
        ]
    )
    offsets = torch.tensor(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0]

        ],
        dtype=torch.int32
    )
    offsets_real = offsets.to(torch.float32)
    dist, edges, angle, triplets = crystal_to_dat(pos, edges, offsets, offsets_real)
    print(dist)
    print(edges)
    print(angle)
    print(triplets)
