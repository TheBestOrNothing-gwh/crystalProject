import numpy as np
from collections import Counter

from pymatgen.core.structure import Structure
from yaff import System, log
log.set_level(0)
from yaff.pes.ext import Cell
from molmod import MolecularGraph
from molmod.units import angstrom
from toponetx.classes import CombinatorialComplex

from crystalproject.data.prepare.process.graph_match import get_linkages, get_bond_linkages, get_isolated_partitions
from crystalproject.data.prepare.process.check import check_isolated, check_period_connection, check_valence


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


def create_crystal_topo(cif_path, radius=8.0, max_num_nbr=12, use_bond_types=False, bond_types=[], linker_types=[]):
    structure = Structure.from_file(cif_path)
    rvecs = structure.lattice._matrix
    numbers = np.array(structure.atomic_numbers)
    pos = structure.cart_coords
    # 转换为原子单位制
    cell = Cell(rvecs)
    frac_pos = np.dot(pos, cell.gvecs.T)
    rvecs = rvecs * angstrom
    pos = np.dot(frac_pos, rvecs)
    system = System(pos=pos, numbers=numbers, rvecs=rvecs)
    if system.bonds is None:
        system.detect_bonds()
    # 合法性检查
    assert check_period_connection(system), "错误的周期性边界条件导致晶格间不连通"
    assert check_valence, "结构中存在错误的化合价，如氢原子形成了两个键等"
    check_result, system = check_isolated(system)
    assert check_result, "结构中存在游离的片段"
    
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
    
    return {
        "atom_radius_graph": atom_radius_graph,
        "atom_graph": atom_graph,
        "cluster_graph": cluster_graph,
        "underling_network": underling_network,
    }


if __name__ == "__main__":
    data = create_crystal_topo("/home/gwh/project/crystalProject/DATA/cofs_Methane/error/linker100_NH_linker7_CH2_ukg_relaxed.cif",
                               use_bond_types=True,
                               bond_types=["amine"],
                               linker_types=[])
    print(-1 in data["atom_graph"]["offsets"])
    print(data["cluster_graph"]["inter"])
    print(data["cluster_graph"]["edges"])
    print(data["cluster_graph"]["offsets"])
    print(data["underling_network"]["edges"])
    print(data["underling_network"]["pos"])
    print(data["underling_network"]["offsets"])