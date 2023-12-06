# MOFTransformer version 2.0.0
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
import seaborn as sns
from ase.data import covalent_radii
from molmod.units import angstrom

from crystalproject.assets.colors import cpk_colors
from crystalproject.visualize.utils import plot_cube
from crystalproject.visualize.setting import get_fig_ax


def draw_colorbar(fig, ax, cmap, minatt, maxatt, **cbar_kwargs):
    norm = Normalize(0.0, 1.0)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        smap, ax=ax, fraction=cbar_kwargs["fraction"], shrink=cbar_kwargs["shrink"]
    )
    cbar.ax.tick_params(labelsize=cbar_kwargs["fontsize"])
    ticks_loc = np.linspace(0, 1, cbar_kwargs["num_ticks"])
    ticks_label = np.round(
        np.linspace(minatt, maxatt, cbar_kwargs["num_ticks"]),
        decimals=cbar_kwargs["decimals"],
    )
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cbar.ax.set_yticklabels(ticks_label)

    cbar.ax.set_ylabel(
        "Attention score",
        rotation=270,
        labelpad=cbar_kwargs["labelpad"],
        fontdict={"size": cbar_kwargs["labelsize"]},
    )


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
    vec1, vec2, vec3 = lattice[0], lattice[1], lattice[2]
    if s_point is None:
        s_point = np.zeros(3)

    opp_vec = vec1 + vec2 + vec3 + s_point

    for v1, v2 in combinations([vec1, vec2, vec3], 2):
        draw_line(ax, s_point, s_point + v1, **kwargs)
        draw_line(ax, s_point, s_point + v2, **kwargs)
        draw_line(ax, s_point + v1, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v2, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v1 + v2, opp_vec, **kwargs)


def draw_atoms(ax, atoms):
    """
    Draw p_atoms using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param atoms: <ase.p_atoms> Target p_atoms for drawing
    :param atomic_scale: <float> scaling factor for draw_atoms.
    """
    coords = atoms["pos"]
    atomic_numbers = atoms["numbers"]
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers]) * angstrom
    atomic_colors = np.array([cpk_colors[i] for i in atomic_numbers])
    ax.scatter(
        xs=coords[:, 0],
        ys=coords[:, 1],
        zs=coords[:, 2],
        c=atomic_colors,
        s=atomic_sizes,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=1.0,
    )
    

def draw_heatmap_grid(ax, positions, colors, lattice, num_patches, alpha=0.5, **kwargs):
    cubes = plot_cube(
        positions,
        colors,
        lattice=lattice,
        num_patches=num_patches,
        edgecolor=None,
        alpha=alpha,
    )
    ax.add_collection3d(cubes, **kwargs)


def draw_heatmap_graph(ax, atoms, uni_idx, colors, atomic_scale, alpha):
    coords = atoms.get_positions()
    for i, idxes in enumerate(uni_idx):
        uni_coords = coords[idxes]
        # att = heatmap_graph[i]
        # c = cmap(scaler(att, minatt, maxatt))
        ax.scatter(
            xs=uni_coords[:, 0],
            ys=uni_coords[:, 1],
            zs=uni_coords[:, 2],
            color=colors[i],
            s=atomic_scale,
            marker="o",
            linewidth=0,
            alpha=alpha,
        )


def draw_compare(fig, ax, x, y, x_label, y_label, addition, title):
    # 绘制 hex bin
    hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap="BuGn")
    # 添加colorbar
    fig.colorbar(hb, ax=ax, label="log10(N)")
    # 添加title
    ax.set_title(title, fontsize=18, fontfamily="sans-serif", fontstyle="italic")
    # 添加label
    ax.set_xlabel(x_label, fontsize=12, fontfamily="sans-serif", fontstyle="italic")
    ax.set_ylabel(y_label, fontsize=12, fontfamily="sans-serif", fontstyle="italic")
    # 添加对角直线
    Axis_line = np.linspace(*ax.get_xlim(), 2)
    ax.plot(Axis_line, Axis_line, transform=ax.transAxes, linestyle="--", linewidth=1, color="red")
    # 添加附加信息，最左上角添加说明
    ax.text(
        ax.get_xlim()[0] * 0.95 + ax.get_xlim()[1] * 0.05, 
        ax.get_ylim()[0] * 0.2 + ax.get_ylim()[1] * 0.8, 
        addition, 
        fontsize=12, 
        fontfamily="sans-serif",
        fontstyle="italic",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.7
        }
    )

def draw_topo(ax, topo):
    # 画出晶格
    draw_cell(ax, topo["atom_graph"]["rvecs"], color="black")
    # 画出原子图
    draw_atoms(ax, topo["atom_graph"], atomic_scale=25)
    for i, (pos1, pos2) in enumerate(topo["atom_graph"]["pos"][topo["atom_graph"]["edges"].T]):
        offset = topo["atom_graph"]["offsets"][i]
        offset_real = topo["atom_graph"]["offsets_real"][i]
        if any(offset != 0.):
            draw_line(ax, pos1, pos2+offset_real, color="black", linewidth=0.5, alpha=0.2)
        else:
            draw_line(ax, pos1, pos2, color="black", linewidth=0.5, alpha=0.1)
    # 画出拓扑图
    ax.scatter(
        xs=topo["underling_network"]["pos"][:, 0],
        ys=topo["underling_network"]["pos"][:, 1],
        zs=topo["underling_network"]["pos"][:, 2],
        c="darkblue",
        s=500.,
        alpha=0.2
    )
    for i, (pos1, pos2) in enumerate(topo["underling_network"]["pos"][topo["underling_network"]["edges"].T]):
        offset = topo["underling_network"]["offsets"][i]
        offset_real = topo["underling_network"]["offsets_real"][i]
        if any(offset != 0.):
            draw_line(ax, pos1, pos2+offset_real, color="red", linestyle="--", linewidth=3, alpha=0.2)
        else:
            draw_line(ax, pos1, pos2, color="red", linestyle="-", linewidth=3, alpha=0.2)    

