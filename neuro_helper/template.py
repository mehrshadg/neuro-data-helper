import numpy as np
import cifti
from pandas import DataFrame, Series
from neuro_helper.entity import TopoName, Space, TemplateName
from neuro_helper.assets import manager

_loaded_templates = {}

file_names = {
    TopoName.MEDIAL_WALL: {
        Space.K32: None,
        Space.K32_CORTEX: "topo.medial_wall.32k_cortex.dlabel.nii",
        Space.K59: "topo.medial_wall.59k.dlabel.nii",
        Space.K59_CORTEX: "topo.medial_wall.59k_cortex.dlabel.nii"
    },
    TopoName.T1T2: {
        Space.K32: None,
        Space.K32_CORTEX: "topo.t1t2.32k_cortex.dscalar.nii",
        Space.K59: None,
        Space.K59_CORTEX: "topo.t1t2.59k_cortex.dscalar.nii",
    },
    TopoName.ANT_POST_GRADIENT: {
        Space.K32: "topo.coord.32k.dscalar.nii",
        Space.K32_CORTEX: "topo.coord.32k_cortex.dscalar.nii",
        Space.K59: None,
        Space.K59_CORTEX: None,
    },
    TopoName.MARGULIES_GRADIENT: {
        Space.K32: "topo.margulies2016.32k.dscalar.nii",
        Space.K32_CORTEX: "topo.margulies2016.32k_cortex.dscalar.nii",
        Space.K59: "topo.margulies2016.59k.dscalar.nii",
        Space.K59_CORTEX: "topo.margulies2016.59k_cortex.dscalar.nii",
    },
    TemplateName.SCHAEFER_200_7: {
        Space.K32: None,
        Space.K32_CORTEX: "template.schaefer2018.2007.32k_cortex.dlabel.nii",
        Space.K59: None,
        Space.K59_CORTEX: "template.schaefer2018.2007.59k_cortex.dlabel.nii",
    },
    TemplateName.SCHAEFER_200_17: {
        Space.K32: None,
        Space.K32_CORTEX: "template.schaefer2018.20017.32k_cortex.dlabel.nii",
        Space.K59: None,
        Space.K59_CORTEX: None,
    },
    TemplateName.COLE_360: {
        Space.K4: None,
        Space.K4_CORTEX: "template.cole.36012.4k_cortex.dlabel.nii",
        Space.K32: "template.cole.36012.32k.dlabel.nii",
        Space.K32_CORTEX: "template.cole.36012.32k_cortex.dlabel.nii",
        Space.K59: "template.cole.36012.59k.dlabel.nii",
        Space.K59_CORTEX: "template.cole.36012.59k_cortex.dlabel.nii"
    },
    TemplateName.WANG: {
        Space.K32: "template.wang2015.32k.dlabel.nii",
        Space.K32_CORTEX: "template.wang2015.32k_cortex.dlabel.nii",
        Space.K59: "template.wang2015.59k.dlabel.nii",
        Space.K59_CORTEX: "template.wang2015.59k_cortex.dlabel.nii"
    }
}


def get_full_path(file_name):
    return manager.get_full_path(manager.AssetCategory.HCP1200, file_name)


def _get_or_load(key, loaded):
    if key not in _loaded_templates:
        if callable(loaded):
            _loaded_templates[key] = loaded()
        else:
            _loaded_templates[key] = loaded
    return _loaded_templates[key]


def get_topo_dataframe(topo_name: TopoName, template_name: TemplateName, space: Space):
    if topo_name == TopoName.T1T2:
        return _get_t1t2_topo(template_name, space)
    elif topo_name == TopoName.MARGULIES_GRADIENT:
        return _get_gradient_topo(template_name, space)
    elif topo_name == TopoName.ANT_POST_GRADIENT:
        return _get_coordinates_topo(template_name, space)
    elif topo_name == TopoName.MEDIAL_WALL:
        return _get_medial_wall_topo(space)
    else:
        raise ValueError(f"{topo_name} is not defined")


def _get_medial_wall_topo(space: Space):
    return _get_or_load(f"{TopoName.MEDIAL_WALL}:{space}",
                        lambda: cifti.read(get_full_path(file_names[TopoName.MEDIAL_WALL][space]))[0].squeeze())


def _get_t1t2_topo(template_name: TemplateName, space: Space):
    def load():
        voxels = cifti.read(get_full_path(file_names[TopoName.T1T2][space]))[0].squeeze()
        mask, _, networks, regions, _ = get_template(template_name, space)
        mask_no_wall = mask[_get_medial_wall_topo(space) == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "t1t2": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        return topo

    return _get_or_load(f"{TopoName.T1T2}:{template_name}:{space}", load)


def _get_gradient_topo(template_name: TemplateName, space: Space):
    def load():
        mask, _, networks, regions, _ = get_template(template_name, space)
        voxels = cifti.read(get_full_path(file_names[TopoName.MARGULIES_GRADIENT][space]))[0].squeeze()
        mask_no_wall = mask[_get_medial_wall_topo(space) == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "gradient": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        return topo

    return _get_or_load(f"{TopoName.MARGULIES_GRADIENT}:{template_name}:{space}", load)


def _get_coordinates_topo(template_name: TemplateName, space: Space):
    def load():
        mask, _, networks, regions, _ = get_template(template_name, space)
        voxels = cifti.read(get_full_path(file_names[TopoName.ANT_POST_GRADIENT][space]))[0].T
        mask_no_wall = mask[_get_medial_wall_topo(space) == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str),
                          "coord_x": Series(dtype=float), "coord_y": Series(dtype=float),
                          "coord_z": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            x, y, z = voxels[mask_no_wall == i + 1, :].mean(axis=0)
            topo.loc[i, :] = reg, net, x, y, z
        return topo

    return _get_or_load(f"{TopoName.ANT_POST_GRADIENT}:{template_name}:{space}", load)


def parcellate(tpt_name: TemplateName, space: Space, voxels):
    mask, _, networks, regions, _ = get_template(tpt_name, space)

    if voxels.shape[0] > mask.size:
        print("INFO: mask is smaller than data. Probably sub-cortex is in the data."
              " Padding mask with 0s (assuming bg=0)")
        mask = np.append(mask, np.zeros(voxels.shape[0] - mask.size))
    elif voxels.shape[0] < mask.size:
        raise ValueError("Mask size is smaller than data size. Something is fishy here!")

    output = np.zeros((len(regions), voxels.shape[1]))
    for ri in range(len(regions)):
        output[ri] = voxels[mask == ri + 1].mean()
    return output


def get_template(name: TemplateName, space: Space):
    return _loaded_templates.get(f"{name}:{space}")


def load_schaefer_template(space: Space, reg_count, net_count):
    if reg_count == 200 and net_count == 7:
        tpt_name = TemplateName.SCHAEFER_200_7
    elif reg_count == 200 and net_count == 17:
        tpt_name = TemplateName.SCHAEFER_200_17
    else:
        raise ValueError(f"SCHAEFER with {reg_count} regions and {net_count} networks is not defined")

    name = f"{tpt_name}:{space}"
    if name not in _loaded_templates:
        mask, (lbl_axis, brain_axis) = \
            cifti.read(get_full_path(file_names[tpt_name][space]))
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[key][0] for key in list(lbl_dict.keys())])[1:]
        networks = [x.split("_")[2] for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name


def load_cole_template(space: Space):
    name = f"{TemplateName.COLE_360}:{space}"
    if name not in _loaded_templates:
        mask, (lbl_axis, brain_axis) = cifti.read(get_full_path(file_names[TemplateName.COLE_360][space]))
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
        networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name


def load_wang_template(space: Space):
    name = f"{TemplateName.WANG}:{space}"
    if name not in _loaded_templates:
        mask, (lbl_axis, brain_axis) = cifti.read(get_full_path(file_names[TemplateName.COLE_360][space]))
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
        networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name


def get_net(net_lbl, template_name: TemplateName):
    if template_name == TemplateName.COLE_360:
        if net_lbl == "pce":
            return {"P": ["Visual1", "Visual2", "Auditory", "Somatomotor"],
                    "EC": ["DorsalAttention", "PosteriorMultimodal", "VentralMultimodal", "OrbitoAffective",
                           "Language", "CinguloOpercular", "Frontoparietal", "Default"]}
        elif net_lbl == "pcr":
            return {"P": ["Visual1", "Visual2", "Auditory", "Somatomotor"],
                    "RC": ["CinguloOpercular", "Frontoparietal", "Default"]}
    elif template_name == TemplateName.SCHAEFER_200_7:
        if net_lbl == "pc":
            return {"P": ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn'],
                    "C": ['Limbic', 'Cont', 'Default']}

    raise ValueError(f"{template_name} and {net_lbl} is not defined")


def net_order(template_name: TemplateName):
    if template_name == TemplateName.COLE_360:
        return ["Visual1", "Visual2", "Auditory", "Somatomotor", "DorsalAttention", "PosteriorMultimodal",
                "VentralMultimodal", "OrbitoAffective", "Language", "CinguloOpercular", "Frontoparietal", "Default"]
    elif template_name == TemplateName.SCHAEFER_200_7:
        return ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    raise Exception(f"{template_name} not defined")


def net_labels(tpt_name: TemplateName, two_line=True):
    if tpt_name == TemplateName.COLE_360:
        names = ['Visual1', 'Visual2', 'Auditory', 'Somatomotor', 'Dorsal\nAttention', 'Posterior\nMultimodal',
                 'Ventral\nMultimodal', 'Orbito\nAffective', 'Language', 'Cingulo\nOpercular', 'FPC', 'DMN']
    elif tpt_name == TemplateName.SCHAEFER_200_7:
        names = ['Visual', 'Somatomotor', 'Dorsal\nAttention', 'Salience', 'Limbic', 'FPC', 'DMN']
    else:
        raise ValueError(f"{tpt_name} not defined in net_labels")

    return names if two_line else [x.replace("\n", " ") for x in names]
