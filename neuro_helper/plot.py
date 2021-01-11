import collections

import os
import cifti

__all__ = ["paired_colors", "triple_colors", "savefig", "savemap"]


def paired_colors(ret_tuple=False):
    if ret_tuple:
        return (0 / 255, 79 / 255, 255 / 255, 1), (162 / 255, 3 / 255, 37 / 255, 1)
    else:
        return "#004fff", "#A20325"


def triple_colors(ret_tuple=False):
    if ret_tuple:
        return (34 / 255, 45 / 255, 64 / 255, 1), (80 / 255, 54 / 255, 29 / 255, 1), (38 / 255, 62 / 255, 44 / 255, 1)
    else:
        return "#5975a4", "#cc8963", "#5f9e6e"


def savefig(fig, name, bbox_inches="tight", extra_artists=(), low=False,
            transparent=False, directory="figures", ext="svg"):
    dpi = 80 if low else 600
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{name}.{ext}"
    fig.savefig(file_path, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches, bbox_extra_artists=extra_artists)
    return os.getcwd() + os.sep + file_path


def savemap(name, data, brain_mask, axes, directory="figures"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{name}.dtseries.nii"
    # noinspection PyTypeChecker
    if not isinstance(axes, collections.Iterable):
        axes = (axes, )
    cifti.write(file_path, data, axes + (brain_mask,))
    return os.getcwd() + os.sep + file_path
