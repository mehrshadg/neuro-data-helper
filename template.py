import numpy as np
import cifti
from pandas import DataFrame, Series

_loaded_templates = {}


def _get_or_load(key, loaded):
    if key not in _loaded_templates:
        if callable(loaded):
            _loaded_templates[key] = loaded()
        else:
            _loaded_templates[key] = loaded
    return _loaded_templates[key]


def get_topo_dataframe(topo_name, template_name):
    if topo_name == "t1t2":
        return _get_t1t2_topo(template_name)
    elif topo_name == "gradient":
        return _get_gradient_topo(template_name)
    elif topo_name == "coord":
        return _get_coordinates_topo(template_name)
    elif topo_name == "medial_wall":
        return _get_medial_wall_topo()
    else:
        raise ValueError(f"{topo_name} is not defined")


def _get_medial_wall_topo():
    return _get_or_load("medial_wall",
                        lambda: cifti.read("files/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii")[0].squeeze())


def _get_t1t2_topo(template_name):
    def load():
        voxels = cifti.read("files/S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii")[0].squeeze()
        mask, _, networks, regions, _ = get_template(template_name)
        mask_no_wall = mask[_get_medial_wall_topo() == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "t1t2": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        return topo

    return _get_or_load(f"t1t2_{template_name}", load)


def _get_gradient_topo(template_name):
    def load():
        mask, _, networks, regions, _ = get_template(template_name)
        voxels = cifti.read("files/gradient_map_2016.32k.dscalar.nii")[0].squeeze()[:29696 + 29716]
        mask_no_wall = mask[_get_medial_wall_topo() == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "gradient": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        return topo

    return _get_or_load(f"gradient_{template_name}", load)


def _get_coordinates_topo(template_name):
    def load():
        mask, _, networks, regions, _ = get_template(template_name)
        voxels = cifti.read("files/S1200.midthickness_MSMAll.32k_fs_LR.coord.dscalar.nii")[0].T[:29696 + 29716, :]
        mask_no_wall = mask[_get_medial_wall_topo() == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str),
                          "coord_x": Series(dtype=float), "coord_y": Series(dtype=float), "coord_z": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(regions, networks)):
            x, y, z = voxels[mask_no_wall == i + 1, :].mean(axis=0)
            topo.loc[i, :] = reg, net, x, y, z
        return topo

    return _get_or_load(f"coord_{template_name}", load)


def get_template(name):
    return _loaded_templates.get(name)


def load_schaefer_template(reg_count, net_count):
    name = f"sh{reg_count}{net_count}"
    if name not in _loaded_templates:
        mask, (lbl_axis, brain_axis) = \
            cifti.read("files/Schaefer2018_%sParcels_%sNetworks_order.dlabel.nii" % (reg_count, net_count))
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[key][0] for key in list(lbl_dict.keys())])[1:]
        networks = [x.split("_")[2] for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name


def load_cole_template():
    name = "cole"
    if name not in _loaded_templates:
        mask, (lbl_axis, brain_axis) = cifti.read(
            "files/CortexColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii")
        mask = np.squeeze(mask)
        lbl_dict = lbl_axis.label.item()
        regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
        networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
        unique_networks = np.unique(networks)
        _loaded_templates[name] = mask, unique_networks, networks, regions, brain_axis
    return name
