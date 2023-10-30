import numpy as np
from collections import Counter

from pymatgen.core.structure import Structure
from yaff import System
from molmod import MolecularGraph, GraphSearch
from molmod.units import angstrom
from toponetx.classes import CombinatorialComplex

from crystalproject.data.prepare.process.utils import patterns, cc, cn, crit1, crit2


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

def get_linkages(system):
    graph = MolecularGraph(system.bonds, system.numbers)
    n_parts = max(get_isolated_parts(graph)) + 1
    linkages_indices = set([])
    linkages = set()
    for ligand in sorted(patterns.keys(), key=lambda e: len(patterns[e][1]), reverse = True):
        pattern, allowed = patterns[ligand]
        indices = set([])
        # Search for pattern
        graph_search = GraphSearch(pattern)
        new_graph = graph
        while True:
            for match in graph_search(new_graph):
                # 第一个约束
                if ligand == "ccnhccnh":
                    print("asdfasdfasdfasd")
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

def get_bond_linkages(graph):
    # Search for C-C or C-N bond by partitioning the system in SBUs
    all_bonds = set([])
    bond_linkages = set()
    for linkages in [cc, cn]:
        indices = set([])
        all_bonds_linkage = set([])
        for name in sorted(linkages.keys(), key=lambda e: linkages[e][0].pattern_graph.num_vertices, reverse = True):
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
                else:
                    break
        all_bonds.update(all_bonds_linkage)
    bond_linkages = list(bond_linkages)
    return bond_linkages, all_bonds

def divide_graphs(system):
    # 计算得到linkage的序号，然后可以将不在linkage序号中的原子分离出来，得到linker的序号。
    graph = MolecularGraph(system.bonds, system.numbers)
    linkages, linkages_indices = get_linkages(system)
    sub_graph = graph.get_subgraph(
        [i for i in range(graph.num_vertices) if i not in linkages_indices]
    )
    bond_linkages, linkages_bonds = get_bond_linkages(sub_graph)
    graph_edges = list(sub_graph.edges)
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


def create_crystal_topo(cif_path):
    structure = Structure.from_file(cif_path)
    rvecs = structure.lattice._matrix
    numbers = np.array(structure.atomic_numbers)
    pos = structure.cart_coords
    system = System(pos=pos, numbers=numbers, rvecs=rvecs)
    if system.bonds is None:
        system.detect_bonds()
    
    # region计算原子图
    # 将bonds扩充为原来的两倍，并且计算offset，得到edges和offset
    edges = []
    offsets = []
    for ibond in range(len(system.bonds)):
        i0, i1 = system.bonds[ibond]
        delta = system.pos[i0] - system.pos[i1]
        # 转换为分数距离
        frac = np.dot(system.cell.gvecs, delta)
        offset = np.ceil(frac - 0.5)
        edges.append([i0, i1])
        offsets.append(offset)
        edges.append([i1, i0])
        offsets.append(-offset)
    edges, offsets = np.array(edges).T, np.array(offsets)
    atom_graph = {
        "numbers": numbers,
        "edges": edges,
        "pos": pos,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    # endregion
    
    # region 计算粗粒度图，不仅要得到粗粒度图的表示，还需要得到vertex到supervertex的关系矩阵，这部分是关键
    partitions = divide_graphs(system)
    print(partitions)
    cc = CombinatorialComplex()
    cc.add_cells_from(range(system.natom), ranks=0)
    cc.add_cells_from(partitions, ranks=1)
    # 计算关联矩阵
    B01 = cc.incidence_matrix(0, 1).todense()
    # 计算粗粒度图中每个原子簇的重心的坐标
    pos = []
    graph = MolecularGraph(system.bonds, system.numbers)
    atom_bonds, atom_offsets = edges.T, offsets
    for partition in partitions:
        sub_graph = graph.get_subgraph(partition)
        central_vertices = sub_graph.central_vertices
        central_vertex = sub_graph.central_vertex
        pos_tmp = []
        for vertex, length, path in sub_graph.iter_breadth_first(central_vertex, do_paths=True):
            if vertex in central_vertices:
                # 统计当前节点相对于中心原子的offset是多少，根据整条路径不断累加就可以得到了
                offset = np.array([0., 0., 0.])
                for i in range(length):
                    bond = np.array([path[i], path[i+1]])
                    idx = np.argwhere((atom_bonds == bond).all(axis=1)).ravel()[0]
                    offset += atom_offsets[idx]
                pos_tmp.append(system.pos[vertex] + np.dot(offset, rvecs))
        #求出重心坐标后先转换为分数坐标判断处于哪个象限，然后再通过求余预算移动到原始晶格内，最后再根据晶格矢量转换为笛卡尔坐标
        pos.append(np.dot(np.remainder(np.dot(system.cell.gvecs, np.mean(np.array(pos_tmp), axis=0)), 1.), rvecs))
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
    cluster_graph = {
        "incidence_martrix": B01,
        "edges": edges,
        "pos": pos,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    print(edges.T)
    print(pos)
    print(offsets)
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
    indices = np.array(list(set(edges[0])))
    underling_network = {
        "indices": indices,
        "edges": edges,
        "pos": pos,
        "offsets": offsets,
        "rvecs": rvecs,
    }
    print(indices)
    print(edges.T)
    print(pos)
    print(offsets)
    # endregion
    return {
        "atom_graph": atom_graph,
        "cluster_graph": cluster_graph,
        "underling_network": underling_network,
    }

if __name__ == "__main__":
    data = create_crystal_topo("/home/gwh/project/crystalProject/models/test/test_topo/10-30/1039.cif")