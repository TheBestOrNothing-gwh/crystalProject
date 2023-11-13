import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from ase.data import covalent_radii
from crystalproject.data.utils.colors import cpk_colors



def draw_line(ax, pos1, pos2, **kwargs):
    """
    Draw line from position 1 to position 2
    :param ax: <matplotlib.axes> figure axis
    :param pos1: <np.array> starting point position
    :param pos2: <np.array> end point position
    :param kwargs: matplotlib plot3D kwargs
    :return:
    """
    ax.plot3D(*zip(pos1, pos2), **kwargs)


def draw_cell(ax, lattice, s_point=None, **kwargs):
    """
    Draw unit-p_lattice p_lattice using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param lattice: <np.array> p_lattice vectors (3 X 3 matrix)
    :param s_point: <np.array> start point of p_lattice
    :param kwargs: matplotlib plot3D kwargs
    """
    vec1, vec2, vec3 = lattice
    if s_point is None:
        s_point = np.zeros(3)

    opp_vec = vec1 + vec2 + vec3 + s_point

    for v1, v2 in combinations([vec1, vec2, vec3], 2):
        draw_line(ax, s_point, s_point + v1, **kwargs)
        draw_line(ax, s_point, s_point + v2, **kwargs)
        draw_line(ax, s_point + v1, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v2, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v1 + v2, opp_vec, **kwargs)


def draw_atoms(ax, atoms, atomic_scale):
    """
    Draw p_atoms using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param atoms: <ase.p_atoms> Target p_atoms for drawing
    :param atomic_scale: <float> scaling factor for draw_atoms.
    """
    coords = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
    atomic_colors = np.array([cpk_colors[i] for i in atomic_numbers])
    ax.scatter(
        xs=coords[:, 0],
        ys=coords[:, 1],
        zs=coords[:, 2],
        c=atomic_colors,
        s=atomic_sizes * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=1.0,
    )