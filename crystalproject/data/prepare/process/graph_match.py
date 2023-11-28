import numpy as np

from yaff import log
log.set_level(0)
from molmod import MolecularGraph, GraphSearch
from molmod.units import angstrom

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