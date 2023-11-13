import numpy as np
from collections import Counter
import time
from matplotlib import pyplot as plt

from pymatgen.core.structure import Structure
from yaff import System, log
log.set_level(0)
from yaff.pes.ext import Cell
from molmod import MolecularGraph, GraphSearch
from molmod.units import angstrom
from toponetx.classes import CombinatorialComplex

from crystalproject.data.prepare.process.utils import patterns, cc, cn, crit1, crit2
from crystalproject.visualize import *


def get_isolated_parts(graph):
    work = [-1]*graph.num_vertices
    n = 0
    while -1 in work:
        start = work.index(-1)
        for i, _ in graph.iter_breadth_first(start):
            work[i] = n
        n += 1
    return work

def get_isolated_partitions(graph):
    work = get_isolated_parts(graph)
    part_dict = {}
    for i in range(len(work)):
        if work[i] in part_dict:
            part_dict[work[i]].append(i)
        else:
            part_dict[work[i]] = [i]
    partitions = [frozenset(part) for part in part_dict.values() if len(part) > 1]
    return partitions

def get_linkages(system, use_bond_types=False, bond_types=[]):
    graph = MolecularGraph(system.bonds, system.numbers)
    n_parts = max(get_isolated_parts(graph)) + 1
    linkages_indices = set([])
    linkages = set()
    for ligand in sorted(patterns.keys(), key=lambda e: len(patterns[e][1]), reverse = True):
        if use_bond_types:
            if ligand.lower() not in [bond_type.lower() for bond_type in bond_types]:
                continue
        pattern, allowed = patterns[ligand]
        indices = set([])
        # Search for pattern
        graph_search = GraphSearch(pattern)
        new_graph = graph
        while True:
            for match in graph_search(new_graph):
                # 第一个约束
                if ligand in crit1.keys():
                    # Extra criterium 1: exact neighbors
                    # e.g.: pyridine ring with propyl and methyl functionalized
                    # nitrogen and carbon-2 atoms would be detected as imine bond
                    nneighs = crit1[ligand]
                    correct_neighs = []
                    c_sp3_hh = False
                    for i in range(len(nneighs[0])):
                        correct_neighs.append(len(graph.neighbors[match.forward[i]]))
                        if correct_neighs[-1] == 4 and not (ligand == 'Amine' and i == 0):
                            # If there are 4 neighbors, it is meant to allow for adamantane
                            # Errors are obtained when an sp3 carbon with two hydrogens is present
                            # E.g. linker 41 in Martin2014
                            count_h = 0
                            for j in graph.neighbors[match.forward[i]]:
                                if system.numbers[j] == 1:
                                    count_h += 1
                            c_sp3_hh = (count_h > 1)
                    if not correct_neighs in nneighs or c_sp3_hh:
                        continue
                # 第二个约束
                if ligand in crit2.keys():
                    # Extra criterium 2: end carbons are not allowed to connect
                    # This avoids detecting a pyridine as an imine bond
                    not_connected = crit2[ligand]
                    not_connected = [match.forward[i] for i in not_connected]
                    if len(not_connected) == 0:
                        # Check all neighbors of the pattern
                        for i in range(pattern.pattern_graph.num_vertices):
                            if i not in allowed:
                                for j in graph.neighbors[match.forward[i]]:
                                    if j not in match.forward.values():
                                        not_connected.append(j)
                    connect = False
                    for i in range(len(not_connected)):
                        for j in range(i+1, len(not_connected)):
                            if not_connected[j] in graph.neighbors[not_connected[i]]:
                                connect = True
                    if connect:
                        continue
                # 第三个约束
                if ligand.startswith('Pyrimidazole'):
                    # There is overlap in the structure, so that a phenyl hydrogen
                    # of the building block has both a neighbor from its phenyl ring and
                    # from the pyrimidazole linkage
                    h_pos = system.pos[match.forward[11]]
                    c_pos = system.pos[match.forward[7]]
                    if np.linalg.norm(h_pos - c_pos) < 1.0*angstrom:
                        continue
                # 第四个约束
                # Sometimes, molmod returns a wrong match. Check the match to be sure
                correct = True
                for pattern_edge in pattern.pattern_graph.edges:
                    graph_edge = frozenset([match.forward[key] for key in pattern_edge])
                    if not graph_edge in graph.edges:
                        correct = False
                for i in range(pattern.pattern_graph.num_vertices):
                    if not graph.numbers[match.forward[i]] == pattern.pattern_graph.numbers[i]:
                        correct = False
                if not correct:
                    continue
                # 第五个约束
                # Check that the linkage is not yet present
                ligand_index = [match.forward[key] for key in allowed]
                if any([i in linkages_indices for i in ligand_index]):
                    assert all([i in linkages_indices for i in ligand_index]), '{} ({}) already occupied'.format(ligand, match.forward.values())
                    continue
                # 第六个约束
                # Extra criterium: the linkage does not create isolated parts, the framework remains connected
                ligand_index = [match.forward[key] for key in allowed]
                subgraph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in ligand_index], normalize = True)
                parts = get_isolated_parts(subgraph)
                if not n_parts == max(parts) + 1:
                    continue
                # Linkage is accepted
                linkage = set(match.forward.values())
                # 如果模式最终选择的点是所有的点，这会导致linkage和linker之间没有共同包含的点，最后形成不了边。
                if len(match.forward.keys()) == len(allowed):
                    for i in match.forward.keys():
                        for vertex, depth in graph.iter_breadth_first(match.forward[i]):
                            if depth >= 2:
                                break
                            linkage.add(vertex)
                print(ligand)
                linkages.add(frozenset(linkage))
                indices.update(ligand_index)
                new_graph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in indices])
                break
            else:
                break
        linkages_indices.update(indices)
    linkages = list(linkages)
    return linkages, linkages_indices

def get_bond_linkages(graph, use_bond_types=False, linkage_types=[]):
    # Search for C-C or C-N bond by partitioning the system in SBUs
    all_bonds = set([])
    bond_linkages = set()
    for linkages in [cc, cn]:
        indices = set([])
        all_bonds_linkage = set([])
        for name in sorted(linkages.keys(), key=lambda e: linkages[e][0].pattern_graph.num_vertices, reverse = True):
            if use_bond_types:
                if name.lower() not in [linkage_type.lower() for linkage_type in linkage_types]:
                    continue
            pattern, bonds = linkages[name]
            bonds = np.array(bonds)
            # Search for pattern
            graph_search = GraphSearch(pattern)
            new_graph = graph
            while True:
                for match in graph_search(new_graph):
                    # Sometimes, molmod returns a wrong match. Check the match to be sure
                    correct = True
                    for pattern_edge in pattern.pattern_graph.edges:
                        graph_edge = frozenset([match.forward[key] for key in pattern_edge])
                        if not graph_edge in graph.edges:
                            correct = False
                    for i in range(pattern.pattern_graph.num_vertices):
                        if not graph.numbers[match.forward[i]] == pattern.pattern_graph.numbers[i]:
                            correct = False
                    if not correct:
                        continue
                    # Check that the building block is not yet present
                    # To prevent that e.g. phenyl-rings are identified in a trisphenyl-benzene block
                    building_block = [match.forward[i] for i in range(pattern.pattern_graph.num_vertices) if not i in bonds.flatten()]
                    if any([i in indices for i in building_block]):
                        continue
                    # Match accepted
                    indices.update(building_block)
                    new_graph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in indices])
                    for bond in bonds:
                        all_bonds_linkage.update([frozenset([match.forward[i] for i in bond])])
                        bond_linkages.add(frozenset([match.forward[i] for i in bond]))
                    break
                else:
                    break
        all_bonds.update(all_bonds_linkage)
    bond_linkages = list(bond_linkages)
    return bond_linkages, all_bonds

def divide_graphs(system, use_bond_types=False, bond_types=[], linker_types=[]):
    # 计算得到linkage的序号，然后可以将不在linkage序号中的原子分离出来，得到linker的序号。
    graph = MolecularGraph(system.bonds, system.numbers)
    linkages, linkages_indices = get_linkages(system, use_bond_types, bond_types)
    sub_graph = graph.get_subgraph(
        [i for i in range(graph.num_vertices) if i not in linkages_indices]
    )
    graph_edges = list(sub_graph.edges)
    bond_linkages, linkages_bonds = get_bond_linkages(sub_graph, use_bond_types, linker_types)
    for bond in linkages_bonds:
        graph_edges.remove(bond)
    linker_graph = MolecularGraph(graph_edges, system.numbers)
    linkers = get_isolated_partitions(linker_graph)
    # 做完划分
    partitions = []
    partitions.extend(linkages)
    partitions.extend(bond_linkages)
    partitions.extend(linkers)
    return partitions


def create_crystal_topo(cif_path, radius=8.0, max_num_nbr=12, use_bond_types = False, bond_types=[], linker_types=[]):
    start = time.time()
    structure = Structure.from_file(cif_path)
    print(time.time() - start)
    start = time.time()
    rvecs = structure.lattice._matrix
    numbers = np.array(structure.atomic_numbers)
    pos = structure.cart_coords
    # 转换为原子单位制
    cell = Cell(rvecs)
    frac_pos = np.dot(pos, cell.gvecs.T)
    rvecs = rvecs * angstrom
    pos = np.dot(frac_pos, rvecs)
    system = System(pos=pos, numbers=numbers, rvecs=rvecs)
    print(time.time() - start)
    start = time.time()
    if system.bonds is None:
        system.detect_bonds()
    
    print(time.time() - start)
    start = time.time()
    # region 计算原子半径图
    sources, targets, offsets, distances = structure.get_neighbor_list(r=radius)
    sources_2, targets_2, offsets_2 = [], [], []
    # 选择最近的一批邻居，至多max_num_nbr个
    for i in range(numbers.shape[0]):
        index = sources == i
        source = sources[index]
        target = targets[index]
        offset = offsets[index]
        distance = distances[index]

        index = np.argsort(distance)
        index = index if index.shape[0] < max_num_nbr else index[:max_num_nbr]
        
        source = source[index]
        target = target[index]
        offset = offset[index]
        distance = distance[index]
        
        sources_2.append(source)
        targets_2.append(target)
        offsets_2.append(offset)
    edges = np.array([np.concatenate(sources_2, axis=0), np.concatenate(targets_2, axis=0)])
    offsets = np.concatenate(offsets_2, axis=0)
    offsets_real = np.dot(offsets, rvecs)
    atom_radius_graph = {
        "numbers": numbers,
        "edges": edges,
        "pos": pos,
        "offsets_real": offsets_real,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    # endregion
    print(time.time() - start)
    start = time.time()
    # region计算原子图
    # 将bonds扩充为原来的两倍，并且计算offset，得到edges和offset
    edges = []
    offsets = []
    for ibond in range(len(system.bonds)):
        i0, i1 = system.bonds[ibond]
        delta = system.pos[i0] - system.pos[i1]
        # 转换为分数距离
        frac = np.dot(delta, system.cell.gvecs.T)
        offset = np.ceil(frac - 0.5)
        edges.append([i0, i1])
        offsets.append(offset)
        edges.append([i1, i0])
        offsets.append(-offset)
    edges, offsets = np.array(edges).T, np.array(offsets)
    offsets_real = np.dot(offsets, rvecs)
    atom_graph = {
        "numbers": numbers,
        "edges": edges,
        "pos": pos,
        "offsets_real": offsets_real,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    # endregion
    print(time.time() - start)
    start = time.time()
    # region 计算粗粒度图，不仅要得到粗粒度图的表示，还需要得到vertex到supervertex的关系矩阵，这部分是关键
    partitions = divide_graphs(system, use_bond_types, bond_types, linker_types)
    cc = CombinatorialComplex()
    cc.add_cells_from(range(system.natom), ranks=0)
    cc.add_cells_from(partitions, ranks=1)
    # 计算粗粒度和原子之间的关联矩阵，使用稀疏矩阵表示
    inter = cc.incidence_matrix(0, 1).tocoo()
    inter = np.array([inter.row, inter.col])
    # 计算粗粒度图中每个原子簇的重心的坐标
    pos = []
    graph = MolecularGraph(system.bonds, system.numbers)
    atom_bonds, atom_offsets = edges.T, offsets
    for partition in partitions:
        sub_graph = graph.get_subgraph(list(partition), normalize=True)
        central_vertices = list(sub_graph._old_vertex_indexes[sub_graph.central_vertices])
        central_vertex = sub_graph._old_vertex_indexes[sub_graph.central_vertex]
        pos_tmp = []
        count = len(central_vertices)
        # 修改为在原图上进行广度优先搜索，这样得到的边的序号是原先的，并且搜索到所有的中心节点后就可以退出了。
        for vertex, length, path in graph.iter_breadth_first(central_vertex, do_paths=True):
            if vertex in central_vertices:
                count -= 1
                # 统计当前节点相对于中心原子的offset是多少，根据整条路径不断累加就可以得到了
                offset = np.array([0., 0., 0.])
                for i in range(length):
                    bond = np.array([path[i], path[i+1]])
                    idx = np.argwhere((atom_bonds == bond).all(axis=1)).ravel()[0]
                    offset += atom_offsets[idx]
                pos_tmp.append(system.pos[vertex] + np.dot(offset, rvecs))       
                if count == 0:
                    break
        #求出重心坐标后先转换为分数坐标判断处于哪个象限，然后再通过求余预算移动到原始晶格内，最后再根据晶格矢量转换为笛卡尔坐标
        pos.append(np.dot(np.remainder(np.dot(np.mean(np.array(pos_tmp), axis=0), system.cell.gvecs.T), 1.), rvecs))
    pos = np.array(pos)
    # 根据重心的坐标计算offset，方法为转换为分数坐标相减后-0.5后再向上取整。这种情况只认为两个点之间只有一条边，对于较大体系，且连边要求的距离较近时对结果不会有影响。
    # 计算粗粒度图的边
    A10 = cc.coadjacency_matrix(1, 0).tocoo()
    # 去除自环
    edges = np.array([[source, target] for source, target in zip(A10.row, A10.col) if source != target]).T
    offsets = []
    for i in range(edges.shape[1]):
        i0, i1 = edges[0, i], edges[1, i]
        delta = pos[i0] - pos[i1]
        # 转换为分数距离
        frac = np.dot(system.cell.gvecs, delta)
        offset = np.ceil(frac - 0.5)
        offsets.append(offset)
    offsets = np.array(offsets)
    offsets_real = np.dot(offsets, rvecs)
    cluster_graph = {
        "inter": inter,
        "edges": edges,
        "pos": pos,
        "offsets_real": offsets_real,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    # endregion
    print(time.time() - start)
    start = time.time()
    # region底层网络图，再粗粒度图上再做一次操作，将所有度为2的superVerte转换为一条边，从而作为底层网络中的边处理。
    # 首先，检索出所有出度为2的点，然后再将这些点去掉，去掉后，原来和这个点相连的两个点之间将会连接。
    pos, edges, offsets = pos, edges, offsets
    while True:
        counter = dict(Counter(edges[0]))
        bond_vertices = [k for k, v in counter.items() if v == 2]
        if len(bond_vertices) == 0:
            # 说明所有出度为2的点都已经被剔除了
            break
        else:
            tmp_edges, tmp_offset = [], []
            # 说明还存在没有处理好的出度为2的点
            bond_vertex = bond_vertices[0]
            source2vertex = np.argwhere(edges[1]==bond_vertex).flatten()
            vertex2target = np.argwhere(edges[0]==bond_vertex).flatten()
            for i in range(len(source2vertex)):
                source = edges[0][source2vertex[i]]
                for j in range(len(vertex2target)):
                    target = edges[1][vertex2target[j]]
                    if (source == target) and (offsets[source2vertex[i]] == -offsets[vertex2target[j]]).all():
                        # 检查是不是同一条边的两种表示，可以通过检查source和target，如果相同，还可以检查offset
                        continue
                    else:
                        tmp_edges.append([source, target])
                        tmp_offset.append(offsets[source2vertex[i]] + offsets[vertex2target[j]])
            # 对edges和offset进行清理，
            indices = np.concatenate((source2vertex, vertex2target), axis=0)
            edges = np.delete(edges, indices, axis=1)
            offsets = np.delete(offsets, indices, axis=0)
            edges = np.concatenate((edges, np.array(tmp_edges).T), axis=1)
            offsets = np.concatenate((offsets, np.array(tmp_offset)), axis=0)
    # 对剩余的节点进行提取，因为此时edges和offsets已经是新的了，但是pos还是没有删除出度为2的点的情况，
    # 只需要提取一下剩余节点的索引，用于最终的readout就可以了。
    indices = np.array(sorted(list(set(edges[0]))))
    pos = pos[indices]
    map = {item:index for index, item in enumerate(indices)}
    edges = np.vectorize(map.get)(edges)
    inter = np.array([[i, j] for i, j in map.items()]).T
    offsets_real = np.dot(offsets, rvecs)
    underling_network = {
        "inter": inter,
        "edges": edges,
        "pos": pos,
        "offsets_real": offsets_real,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    # endregion
    print(time.time() - start)
    start = time.time()
    
    return {
        "atom_radius_graph": atom_radius_graph,
        "atom_graph": atom_graph,
        "cluster_graph": cluster_graph,
        "underling_network": underling_network,
    }

if __name__ == "__main__":
    start = time.time()
    data = create_crystal_topo("/home/bachelor/gwh/project/crystalProject/DATA/cofs_Methane/debug/linker97_C_linker64_C_nbo_relaxed_interp_3.cif",
                               use_bond_types=True,
                               bond_types=["CC"],
                               linker_types=["linker97", "linker64"])
    print(time.time() - start)
    fig, ax = get_fig_ax()
    draw_cell(ax, data["atom_graph"]["rvecs"], color="black")
    draw_atoms(ax, )
    plt.savefig("cell.png")

    print(-1 in data["atom_graph"]["offsets"])
    print(data["cluster_graph"]["inter"])
    print(data["cluster_graph"]["edges"])
    print(data["cluster_graph"]["offsets"])
    print(data["underling_network"]["edges"])
    print(data["underling_network"]["pos"])
    print(data["underling_network"]["offsets"])