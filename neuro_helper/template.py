import numpy as np
import cifti
from pandas import DataFrame, Series
from neuro_helper.entity import TopoName, Space, TemplateName

_loaded_templates = {}

files = {
    TopoName.MEDIAL_WALL: {
        Space.K8: "",
        Space.K32: "files/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii",
        Space.K64: ""
    },
    TopoName.T1T2: {
        Space.K8: "",
        Space.K32: "files/S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii",
        Space.K64: ""
    },
    TopoName.ANT_POST_GRADIENT: {
        Space.K8: "",
        Space.K32: "files/S1200.midthickness_MSMAll.32k_fs_LR.coord.dscalar.nii",
        Space.K64: ""
    },
    TopoName.MARGULIES_GRADIENT: {
        Space.K8: "",
        Space.K32: "files/gradient_map_2016.32k.dscalar.nii",
        Space.K64: ""
    },
    TemplateName.SCHAEFER_200_7: {
        Space.K32: "files/Schaefer2018_200Parcels_7Networks_order.dlabel.nii",
        Space.K64: ""
    },
    TemplateName.SCHAEFER_200_17: {
        Space.K32: "",
        Space.K64: ""
    },
    TemplateName.COLE_360: {
        Space.K32: "files/CortexColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii",
        Space.K64: ""
    },
    TemplateName.WANG: {
        Space.K32: "",
        Space.K64: "files/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii"
    }
}


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
                        lambda: cifti.read(files[TopoName.MEDIAL_WALL][space])[0].squeeze())


def _get_t1t2_topo(template_name: TemplateName, space: Space):
    def load():
        voxels = cifti.read(files[TopoName.T1T2][space])[0].squeeze()
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
        voxels = cifti.read(files[TopoName.MARGULIES_GRADIENT][space])[0].squeeze()[:29696 + 29716]
        mask_no_wall = mask[_get_medial_wall_topo(space) == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "gradient": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        return topo

    return _get_or_load(f"{TopoName.MARGULIES_GRADIENT}:{template_name}:{space}", load)


def _get_coordinates_topo(template_name: TemplateName, space: Space):
    def load():
        mask, _, networks, regions, _ = get_template(template_name, space)
        voxels = cifti.read(files[TopoName.ANT_POST_GRADIENT][space])[0].T[:29696 + 29716, :]
        mask_no_wall = mask[_get_medial_wall_topo(space) == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str),
                          "coord_x": Series(dtype=float), "coord_y": Series(dtype=float),
                          "coord_z": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            x, y, z = voxels[mask_no_wall == i + 1, :].mean(axis=0)
            topo.loc[i, :] = reg, net, x, y, z
        return topo

    return _get_or_load(f"{TopoName.ANT_POST_GRADIENT}:{template_name}:{space}", load)


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
            cifti.read(files[tpt_name][space])
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
        mask, (lbl_axis, brain_axis) = cifti.read(files[TemplateName.COLE_360][space])
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
        networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name
