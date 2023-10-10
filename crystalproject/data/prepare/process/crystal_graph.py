import numpy as np
from pymatgen.core.structure import Structure


def create_crystal_graph(cif_path, radius=8.0, max_num_nbr=12):
    crystal = Structure.from_file(cif_path)
    atom_fea = np.array(
        [
            crystal[i].specie.number - 1
            for i in range(len(crystal))
        ]
    )
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for i, nbr in enumerate(all_nbrs):
        nbr_fea_idx_i = list(map(lambda x: [i, x[2]], nbr))
        nbr_fea_i = list(map(lambda x: [x[1]], nbr))
        nbr_fea_idx.extend(nbr_fea_idx_i if len(nbr_fea_idx_i) < max_num_nbr else nbr_fea_idx_i[:max_num_nbr])
        nbr_fea.extend(nbr_fea_i if len(nbr_fea_i) < max_num_nbr else nbr_fea_i[:max_num_nbr])
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx).T, np.array(nbr_fea)
    return {
        "atom_fea": atom_fea,
        "nbr_fea": nbr_fea,
        "nbr_fea_idx": nbr_fea_idx
    }
