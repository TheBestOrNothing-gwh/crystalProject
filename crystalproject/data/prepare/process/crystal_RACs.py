import numpy as np
import os

from pymatgen.core.structure import Structure
from yaff import System, log
log.set_level(0)
from yaff.pes.ext import Cell
from molmod import MolecularGraph, GraphError
from molmod.periodic import periodic as pt
from molmod.units import angstrom

from crystalproject.data.prepare.process.utils import endict, poldict
from crystalproject.data.prepare.process.graph_match import get_linkages, get_bond_linkages
from crystalproject.data.prepare.process.check import check_isolated, check_period_connection, check_valence


def iter_graphs(system, use_bond_types=False, bond_types=[], linker_types=[]):
    graph = MolecularGraph(system.bonds, system.numbers)
    _, ligands = get_linkages(system, use_bond_types, bond_types)
    sub_graph = graph.get_subgraph(
        [i for i in range(graph.num_vertices) if i not in ligands]
    )
    graph_edges = list(sub_graph.edges)
    _, linkages_bonds = get_bond_linkages(sub_graph, use_bond_types, linker_types)
    for bond in linkages_bonds:
        graph_edges.remove(bond)
    if len(ligands) == 0 and len(linkages_bonds) == 0:
        raise RuntimeError("没有识别到任何反应位点")
    linker_graph = MolecularGraph(graph_edges, system.numbers)
    linkers = set([i for i in range(graph.num_vertices) if i not in ligands])
    connecting = set([i for i in linkers if not len(graph.neighbors[i]) == len(linker_graph.neighbors[i])])
    functional = set([])
    for i0, i1 in linker_graph.edges:
        try:
            part0, part1 = graph.get_halfs(i0, i1)
        except GraphError as err:
            continue
        if any([index in connecting for index in part0]):
            functional_part = list(part1)
        elif any([index in connecting for index in part1]):
            functional_part = list(part0)
        if len(functional_part) == 1 and graph.numbers[functional_part[0]] == 1: continue
        functional.update(functional_part)
    
    yield 'LigandRAC', graph, ligands, set([i for i in range(graph.num_vertices)])
    yield 'FullLinkerRAC', linker_graph, linkers, linkers
    yield 'LinkerConnectingRAC', linker_graph, connecting, linkers
    yield 'FunctionalGroupRAC', linker_graph, functional, linkers

def compute_racs(system, graph, start, scope):
    props = ['I', 'T', 'X', 'S', 'Z', 'a']
    result = {}
    for d in range(4):
        for prop in props:
            for method in ['prod', 'diff']:
                result['_'.join([method, prop, str(d)])] = 0.0
    if len(start) == 0:
        result = np.array(list(result.values()))
        return result, result
    for i in start:
        for j, d in graph.iter_breadth_first(start = i):
            if d == 4: break
            if j not in scope: continue
            for prop in props:
                match prop:
                    case "I":
                        # Identity
                        prop_i = 1
                        prop_j = 1
                    case "T":
                        # Connectivity
                        prop_i = len(graph.neighbors[i])
                        prop_j = len(graph.neighbors[j])
                    case 'X':
                        # Electronegativity
                        prop_i = endict[pt[system.numbers[i]].symbol]
                        prop_j = endict[pt[system.numbers[j]].symbol]
                    case 'S':
                        # Covalent radius
                        # Different definition molmod and molsimplify
                        prop_i = pt[system.numbers[i]].covalent_radius
                        prop_j = pt[system.numbers[j]].covalent_radius
                    case 'Z':
                        # Nuclear charge
                        prop_i = system.numbers[i]
                        prop_j = system.numbers[j]
                    case 'a':
                        # Polarizability
                        prop_i = poldict[pt[system.numbers[i]].symbol]
                        prop_j = poldict[pt[system.numbers[j]].symbol]
                result['_'.join(['diff', prop, str(d)])] += float(prop_i - prop_j)
                result['_'.join(['prod', prop, str(d)])] += float(prop_i*prop_j)
    result = np.array(list(result.values()))
    return result / len(start), result


def create_crystal_RACs(cif_path, use_bond_types=False, bond_types=[], linker_types=[]):
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
    check_period_connection(system, os.path.basename(cif_path).split(".")[0])
    check_valence(system, os.path.basename(cif_path).split(".")[0])
    check_isolated(system, os.path.basename(cif_path).split(".")[0])
    # 计算RACs
    racs = {}
    for name, graph, start, scope in iter_graphs(system, use_bond_types, bond_types, linker_types):
        racs[name+"_mean"], racs[name+"_sum"] = compute_racs(system, graph, start, scope)
    return racs