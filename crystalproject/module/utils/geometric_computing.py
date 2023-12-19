import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi


def crystal_to_dat(pos, edges, offsets, offsets_real):
    # Calculate distances. # number of edges
    dist = (pos[edges[1]] - pos[edges[0]] + offsets_real).norm(dim=-1)

    # Calculate angle
    # get triplets
    value = torch.arange(edges.shape[1], device=edges.device)
    tmp = SparseTensor(row=edges[0], col=edges[1], value=value, sparse_sizes=(pos.shape[0], pos.shape[0]))
    tmp = tmp[edges[1]]
    triplets = torch.stack((tmp.storage.row(), tmp.storage.value()), dim=0)
    mask = (
        torch.eq(edges[0, triplets[0]], edges[1, triplets[1]]) & 
        torch.eq(offsets[triplets[0]], -offsets[triplets[1]]).all(dim=1)
    )
    triplets = triplets[:, ~mask]
    # compute angle
    p_jk = -(pos[edges[1, triplets[0]]] - pos[edges[0, triplets[0]]] + offsets_real[triplets[0]])
    p_ji = pos[edges[1, triplets[1]]] - pos[edges[0, triplets[1]]] + offsets_real[triplets[1]]
    a = (p_jk * p_ji).sum(dim=-1)
    b = torch.cross(p_jk, p_ji).norm(dim=-1)
    angle = torch.atan2(b, a)

    # Calculate torsion
    value = torch.arange(triplets.shape[1], device=triplets.device)
    tmp = SparseTensor(row=triplets[0], col=triplets[1], value=value, sparse_sizes=(edges.shape[1], edges.shape[1]))
    tmp = tmp[:, triplets[1]]
    torsion = torch.stack((tmp.storage.col(), tmp.storage.value()), dim=0)
    mask = (
        torch.eq(triplets[0, torsion[0]], triplets[0, torsion[1]])
    )
    torsion = torsion[:, ~mask]
    # compute dihedral_angle
    p_jn = -(pos[edges[1, triplets[0, torsion[0]]]] - pos[edges[0, triplets[0, torsion[0]]]] + offsets_real[triplets[0, torsion[0]]])
    p_jk = -(pos[edges[1, triplets[0, torsion[1]]]] - pos[edges[0, triplets[0, torsion[1]]]] + offsets_real[triplets[0, torsion[1]]])
    p_ji = pos[edges[1, triplets[1, torsion[1]]]] - pos[edges[0, triplets[1, torsion[1]]]] + offsets_real[triplets[1, torsion[1]]]
    plane1 = torch.cross(p_jn, p_ji)
    plane2 = torch.cross(p_jk, p_ji)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = torch.cross(plane1, plane2).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    dihedral_angle = torch.atan2(b, a) # -pi to pi
    dihedral_angle[dihedral_angle<=0] += 2 * pi # 0 to 2pi
    dihedral_angle = scatter(dihedral_angle, torsion[1], reduce="min")
    dihedral_angle = dihedral_angle.resize_(triplets.shape[1])

    return dist, edges, angle, dihedral_angle, triplets


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
