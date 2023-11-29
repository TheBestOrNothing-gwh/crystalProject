import numpy as np

from yaff import log
log.set_level(0)
from molmod import MolecularGraph

from crystalproject.data.prepare.process.graph_match import get_isolated_parts


def check_valence(system, name):
    """
    H最多只有一个邻居
    C最多四个邻居
    """
    graph = MolecularGraph(system.bonds, system.numbers)
    for i in range(graph.num_vertices):
        match system.numbers[i]:
            case 1:
                # H
                assert len(graph.neighbors[i]) <= 1, f"H原子形成了超过1个键: {name}"
            case 5:
                # N
                assert len(graph.neighbors[i]) <= 3, f"N原子形成了超过3个键: {name}"
            case 6:
                # C
                assert len(graph.neighbors[i]) <= 4, f"C原子形成了超过4个键: {name}"
            case 8:
                # O
                assert len(graph.neighbors[i]) <= 2, f"O原子形成了超过2个键: {name}"
            case _:
                continue

def check_isolated(system, name, threshold = 12):
    '''
    threshold: if an isolated part has less than this number of atoms, it considered
    as an isolated molecule and removed. If it is larger than this, it is considered
    as being part of the framework (e.g. 2D layers or 3D-catenated nets are not connected).
        Default = 12: would allow a hcb system with a 3c-triazine ring and no linkers
        Also checked on CURATED database and 2D layers always have at least 15 atoms
    A more decent approach to clean the system is implemented in the preprocessing routine
    '''
    graph = MolecularGraph(system.bonds, system.numbers)
    parts = get_isolated_parts(graph)
    mask = [True]*system.natom
    for i in range(max(parts) + 1):
        n = sum([k == i for k in parts])
        if n < threshold:
            for k in range(len(parts)):
                if i == parts[k]:
                    mask[k] = False
    indices = np.array([i for i in range(system.natom)])[mask]
    system = system.subsystem(indices)
    assert all(mask), f"结构中存在游离的部分: {name}"

def check_period_connection(system, name):
    """
    检查是否存在offset不为零的边
    """
    flag = False
    for ibond in range(len(system.bonds)):
        i0, i1 = system.bonds[ibond]
        delta = system.pos[i0] - system.pos[i1]
        # 转换为分数距离
        frac = np.dot(delta, system.cell.gvecs.T)
        offset = np.ceil(frac - 0.5)
        if not all(np.isclose(offset, 0, rtol=1e-5)):
            # 存在连接不同晶格原子的化学键
            flag = True
    assert flag, f"结构不连续: {name}"
