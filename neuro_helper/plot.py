import os
import cifti
import seaborn as sns

from neuro_helper.entity import TemplateName
from neuro_helper.colormap import cole_data, schaefer7_data

directory = "figures"
sns.set_style("whitegrid")
PC_colors = ["#637687", "#A20325"]
# PC_colors_tuple = [(99 / 256, 118 / 256, 135 / 256, 1), (162 / 255, 3 / 255, 37 / 255, 1)]
PC_colors_tuple = [(0 / 256, 79 / 256, 255 / 256, 1), (162 / 255, 3 / 255, 37 / 255, 1)]
PC_labels = ["Periphery", "Core"]
task_colors = ["#f032e6", "#aaffc3", "#637687"]
task_rest_colors = ["#B89B49"] + task_colors
PMC_labels = ["Periphery", "Intermediate", "Core"]
PMC_colors = ["#5975a4", "#cc8963", "#5f9e6e"]
PMC_colors_tuple = [(34 / 256, 45 / 256, 64 / 256, 1), (80 / 256, 54 / 256, 29 / 256, 1),
                    (38 / 256, 62 / 256, 44 / 256, 1)]
font_scale = 1.1
sns.set(font_scale=font_scale, style="whitegrid")
template_meta_combination = [
    (TemplateName.SCHAEFER_200_7, "pc"),
    (TemplateName.COLE_360, "pce"),
    (TemplateName.COLE_360, "pcr")
]

net_meta_C = {"pc": "C", "pce": "EC", "pcr": "RC"}


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


def make_lh_pallete(palette):
    return [palette[0], ] * 7 + [palette[-1], ] * 5


def net_labels(tpt_name: TemplateName, two_line=True):
    if tpt_name == TemplateName.COLE_360:
        names = ['Visual1', 'Visual2', 'Auditory', 'Somatomotor', 'Dorsal\nAttention', 'Posterior\nMultimodal',
                 'Ventral\nMultimodal', 'Orbito\nAffective', 'Language', 'Cingulo\nOpercular', 'FPC', 'DMN']
    elif tpt_name == TemplateName.SCHAEFER_200_7:
        names = ['Visual', 'Somatomotor', 'Dorsal\nAttention', 'Salience', 'Limbic', 'FPC', 'DMN']
    else:
        raise ValueError(f"{tpt_name} not defined in net_labels")

    return names if two_line else [x.replace("\n", " ") for x in names]
