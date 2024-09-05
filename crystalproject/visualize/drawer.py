# MOFTransformer version 2.0.0
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from ase.data import covalent_radii
from molmod.units import angstrom

from crystalproject.assets.colors import cpk_colors


"""
提供各种基础的绘图函数，后续的visualizer模块用于整体的逻辑整合和绘制。
"""

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
        alpha=1.0,
    )

def draw_bonds(ax, graph, color, linewidth):
    """
    一般来说，原子图的边使用黑色，而拓扑图使用红色。
    """
    for i, (pos1, pos2) in enumerate(graph["pos"][graph["edges"].T]):
        offset = graph["offsets"][i]
        offset_real = graph["offsets_real"][i]
        if any(offset != 0.):
            draw_line(ax, pos1, pos2+offset_real, color=color, linestyle="--", linewidth=linewidth, alpha=0.2)
        else:
            draw_line(ax, pos1, pos2, color=color, linestyle="-", linewidth=linewidth, alpha=0.2)
    
def draw_topo(ax, graph):
    # 画出拓扑图
    ax.scatter(
        xs=graph["pos"][:, 0],
        ys=graph["pos"][:, 1],
        zs=graph["pos"][:, 2],
        c="darkblue",
        s=500.,
        alpha=0.2
    )


# 下面的绘图函数主要用于注意力系数的展示    
def draw_colorbar(fig, ax, cmap, minatt, maxatt, **cbar_kwargs):
    """
    绘制热度图，主要用于存在注意力机制的情况下，进行注意力的展示
    """
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


# 这个函数用于模型预测效果的展示
def draw_compare(fig, ax, x, y, x_label, y_label, addition, title="对比密度图"):
    # 绘制 hex bin
    hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap="BuGn")
    # 添加colorbar
    fig.colorbar(hb, ax=ax, label="log144, 141(N)")
    # 添加title
    ax.set_title(title, fontsize=18, fontfamily="sans-serif", fontstyle="italic")
    # 添加label
    ax.set_xlabel(x_label, fontsize=12, fontfamily="sans-serif", fontstyle="italic")
    ax.set_ylabel(y_label, fontsize=12, fontfamily="sans-serif", fontstyle="italic")
    ax.set_aspect("equal", "box")
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

  

