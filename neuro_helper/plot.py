import os
import cifti

from neuro_helper.colormap import cole_data, schaefer7_data
from neuro_helper.entity import TemplateName

directory = "figures"


def paired_colors(ret_tuple=False):
    if ret_tuple:
        return (0 / 256, 79 / 256, 255 / 256, 1), (162 / 255, 3 / 255, 37 / 255, 1)
    else:
        return "#637687", "#A20325"


def triple_colors(ret_tuple=False):
    if ret_tuple:
        return (34 / 256, 45 / 256, 64 / 256, 1), (80 / 256, 54 / 256, 29 / 256, 1), (38 / 256, 62 / 256, 44 / 256, 1)
    else:
        return "#5975a4", "#cc8963", "#5f9e6e"


def paired_labels(mode: str):
    if mode == "cp":
        return "Periphery", "Core"
    elif mode == "ut":
        return "Unimodal", "Transmodal"
    else:
        raise ValueError(f"No paired labels are defined for {mode}")


def savefig(fig, name, bbox_inches="tight", extra_artists=(), low=False, transparent=False):
    dpi = 80 if low else 600
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{name}.png"
    fig.savefig(file_path, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches, bbox_extra_artists=extra_artists)
    return os.getcwd() + os.sep + file_path


def savemap(name, data, brain_mask, *axes):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{name}.dtseries.nii"
    # noinspection PyTypeChecker
    cifti.write(file_path, data, axes + (brain_mask,))
    return os.getcwd() + os.sep + file_path


def make_net_palette(template_name: TemplateName):
    if template_name == TemplateName.COLE_360:
        return cole_data
    elif template_name == TemplateName.SCHAEFER_200_7:
        return schaefer7_data
    else:
        raise ValueError(f"{template_name} has no network template.")


def net_labels(tpt_name: TemplateName, two_line=True):
    if tpt_name == TemplateName.COLE_360:
        names = ['Visual1', 'Visual2', 'Auditory', 'Somatomotor', 'Dorsal\nAttention', 'Posterior\nMultimodal',
                 'Ventral\nMultimodal', 'Orbito\nAffective', 'Language', 'Cingulo\nOpercular', 'FPC', 'DMN']
    elif tpt_name == TemplateName.SCHAEFER_200_7:
        names = ['Visual', 'Somatomotor', 'Dorsal\nAttention', 'Salience', 'Limbic', 'FPC', 'DMN']
    else:
        raise ValueError(f"{tpt_name} not defined in net_labels")

    return names if two_line else [x.replace("\n", " ") for x in names]
