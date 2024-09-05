# MOFTransformer version 2.0.0

from collections.abc import Sequence
from functools import wraps, partial
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation

from crystalproject.visualize.utils import (
    get_heatmap,
    scaler,
    get_batch_from_cif_name,
    get_model_and_datamodule,
)
from crystalproject.visualize.setting import (
    get_fig_ax,
    set_fig_ax,
    set_axes_equal,
    DEFAULT_FIGSIZE,
    DEFAULT_VIEW_INIT,
    get_default_cbar_kwargs,
    get_cmap,
)
from crystalproject.visualize.drawer import (
    draw_cell,
    draw_atoms,
    draw_bonds,
    draw_topo,
    draw_colorbar,
    draw_heatmap_graph,
)


class Visualizer(object):
    def __init__(self, batch_data, **kwargs):
        """
        Attention Visualizer from "MOFTransformer model"
        :param batch_data: (narray) 只包含一个数据的batch，从里面可以提取各种信息，相当于只有一个结构
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
        """
        self.batch_data = batch_data
        self.kwargs = kwargs
        self.cbar_kwargs = get_default_cbar_kwargs(self.figsize)

    @classmethod
    def from_batch(cls, batch, batch_idx, model, **kwargs):
        """
        Attention visualizer from "MOFTransformer" model and "dataloader batch"
        :param batch: Dataloader -> batch
        :param batch_idx: index for batch index
        :param model: <torch.model> fine-tuned MOFTransformer model
        :param cif_root: <str> root dir for cif files
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
        :return: <PatchVisualizer> patch visualizer object
        """
        model(batch)
        #get_heatmap(batch, batch_idx)
        return cls(
            batch, **kwargs
        )

    @classmethod
    def from_cifname(
        cls, cif_name, config_path, model_path, **kwargs
    ):
        """
        Create PatchVisualizer from cif name. cif must be in test.json or test_{downstream}.json.

        :param config_path: 记录模型和dataset参数的文件路径
        :param kwargs:
            figsize : (float, float) figure size
            view_init : (float, float) view init from matplotlib
            show_axis : <bool> If True, axis are visible. (default : False)
            show_colorbar : <bool> If True, colorbar are visible. (default : True)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
        :return: PatchVisualizer class for index
        """
        model, data_iter = get_model_and_datamodule(config_path, model_path)
        batch = get_batch_from_cif_name(data_iter, cif_name)
        return cls.from_batch(batch, 0, model, **kwargs)

    @property
    def figsize(self):
        return self.kwargs.get("figsize", DEFAULT_FIGSIZE)

    @figsize.setter
    def figsize(self, figsize):
        if not isinstance(figsize, Sequence):
            raise TypeError(f"figsize must be tuple or list, not {type(figsize)}")
        elif len(figsize) != 2:
            raise ValueError(f"figsize must be (float, float) not {figsize}")
        self.kwargs["figsize"] = figsize
        self._sync_cbar_kwargs()

    @property
    def view_init(self):
        return self.kwargs.get("view_init", DEFAULT_VIEW_INIT)

    @view_init.setter
    def view_init(self, view_init):
        if not isinstance(view_init, Sequence):
            raise TypeError(f"view_init must be tuple or list, not {type(view_init)}")
        elif len(view_init) != 2:
            raise ValueError(f"view_init must be (float, float) not {view_init}")
        self.kwargs["view_init"] = view_init

    @property
    def show_axis(self):
        return self.kwargs.get("show_axis", False)

    @show_axis.setter
    def show_axis(self, show_axis):
        if not isinstance(show_axis, bool):
            raise TypeError(f"show_axis must be bool, not {type(show_axis)}")
        self.kwargs["show_axis"] = show_axis

    @property
    def show_colorbar(self):
        return self.kwargs.get("show_colorbar", True)

    @show_colorbar.setter
    def show_colorbar(self, show_colorbar):
        if not isinstance(show_colorbar, bool):
            raise TypeError(f"show_colorbar must be bool, not {type(show_colorbar)}")
        self.kwargs["show_colorbar"] = show_colorbar
    
    def set_colorbar_options(self, default=False, **cbar_kwargs):
        if default:
            self.cbar_kwargs = get_default_cbar_kwargs(self.figsize)
        else:
            for key, value in cbar_kwargs.items():
                if key in self.cbar_kwargs:
                    self.cbar_kwargs[key] = value
    
    @property
    def cmap(self):
        return self.kwargs.get("cmap", None)

    @cmap.setter
    def cmap(self, cmap):
        if isinstance(cmap, (str, ListedColormap)) or cmap is None:
            self.kwargs["cmap"] = cmap
        else:
            raise TypeError(
                f"cmap must be str, ListedColormap, or None, not {type(cmap)}"
            )

    def set_default(self):
        self.kwargs = {}
        self.cbar_kwargs = get_default_cbar_kwargs(self.figsize)
    
    def animate(self, func, frame=360, interval=20, savefile=None, fps=30):
        def turn(i, ax, fig, **kwargs):
            view_init = kwargs.get("view_init", self.view_init)
            ax.view_init(elev=view_init[0], azim=i)
            return fig

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs["return_fig"] = True
            fig, ax = func(*args, **kwargs)
            anim = animation.FuncAnimation(
                fig,
                partial(turn, ax=ax, fig=fig, **kwargs),
                init_func=lambda: fig,
                frames=frame,
                interval=interval,
                blit=True,
            )
            if savefile:
                anim.save(savefile, fps=fps, dpi=300)

            plt.show()
            return anim

        return wrapper

    def draw(
        self,
        minatt=0.000,
        maxatt=0.010,
        alpha=0.7,
        return_fig=True,
        **kwargs,
    ):
        """
        Draw graph attention score figure in primitive unit cell
        :param minatt: (float) Minimum value of attention score (default : 0.000). A value smaller than minatt is treated as minatt.
        :param maxatt: (float) Maximum value of attention score (default : 0.010). A value larger than maxatt is treated as maxatt.
        :param alpha: (float) The alpha blending value, between 0 (transparent) and 1 (opaque).
        :param return_fig : (bool) If True, matplotlib.figure.Figure and matplotlib.Axes3DSubplot are returned.
        :param kwargs:
            view_init : (float, float) view init from matplotlib
            show_colorbar : <bool> If True, colorbar are visible. (default : False)
            cmap : (str or matplotlib.colors.ListedColormap) color map used in figure. (default : None)
        """
        # 构建基本的图层
        fig, ax = get_fig_ax(**self.kwargs)
        # cmap = get_cmap(kwargs.get("cmap", self.cmap))
        set_fig_ax(ax, **kwargs)

        # 绘制晶格
        draw_cell(ax, self.batch_data["atom_bond_graph"]["rvecs"][0], color="black")

        # 绘制原子
        draw_atoms(
            ax, 
            self.batch_data["atom_bond_graph"]
        )
        
        # 绘制原子键
        draw_bonds(
            ax, 
            self.batch_data["atom_bond_graph"],
            "black",
            0.8
        )

        # 绘制底层网络
        draw_topo(
            ax,
            self.batch_data["underling_network"]
        )

        draw_bonds(
            ax,
            self.batch_data["underling_network"],
            "red",
            2.0
        )

        # 可视化注意力的时候才需要添加，原子的注意力使用额外的球表示，而拓扑的点直接使用球的大小表示注意力的情况
        # colors = cmap(scaler(heatmap_graph, minatt, maxatt))
        # draw_heatmap_graph(ax, atoms, self.uni_idx, colors, atomic_scale, alpha)

        # if kwargs.get("show_colorbar", self.show_colorbar):
        #     draw_colorbar(fig, ax, cmap, minatt, maxatt, **self.cbar_kwargs)

        set_axes_equal(ax)
        if return_fig:
            return fig, ax
        else:
            plt.show()

    
