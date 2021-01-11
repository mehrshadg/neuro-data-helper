import numpy as np
import cifti
from pandas import DataFrame, Series
from neuro_helper.abstract.map import *
from neuro_helper.generic import value_or_raise
from neuro_helper.generic import break_space as break_space_func

__all__ = ["MapMemory", "T1T2Topo", "MarguliesGradientTopo", "AntPostTopo",
           "SchaeferTemplateMap", "ColeTemplateMap", "WangTemplateMap"]


class MapMemory:
    _loaded = {}

    def put(self, map_: AbstractMap):
        if isinstance(map_, TemplateMap):
            key = f"{map_.name}:{map_.space}"
        elif isinstance(map_, TopoMap):
            key = f"{map_.name}:{map_.template.name}:{map_.space}"
        else:
            raise NotImplementedError(f"{type(map_)} is not defined in MapMemory")
        self._loaded[key] = map_

    def get_template(self, name: TemplateName, space: Space):
        return value_or_raise(
            self._loaded, f"{name}:{space}",
            f"Template {name} with space {space} is not loaded into memory.")

    def get_topo(self, name: TopoName, tpt_name: TemplateName, space: Space):
        return value_or_raise(
            self._loaded, f"{name}:{tpt_name}:{space}",
            f"Topo {name} for template {tpt_name} with space {space} is not loaded into memory.")

    def __getitem__(self, key):
        return self._loaded[key]

    def __setitem__(self, key, value):
        self._loaded[key] = value


class T1T2Topo(TopoMap):
    def __init__(self, template: TemplateMap):
        super().__init__(TopoName.T1T2, template)

    def load(self):
        if self.loaded:
            return
        self.template()
        voxels = cifti.read(self.file_full_path)[0].squeeze()
        mask_no_wall = self.template.data.mask[self.medial_wall_mask == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "t1t2": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(self.template.data.regions, self.template.data.networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        self._data = topo


class MarguliesGradientTopo(TopoMap):
    def __init__(self, template: TemplateMap):
        super().__init__(TopoName.MARGULIES_GRADIENT, template)

    def load(self):
        if self.loaded:
            return
        self.template()
        voxels = cifti.read(self.file_full_path)[0].squeeze()
        mask_no_wall = self.template.data.mask[self.medial_wall_mask == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str), "gradient": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(self.template.data.regions, self.template.data.networks)):
            topo.loc[i, :] = reg, net, voxels[mask_no_wall == i + 1].mean()
        self._data = topo


class AntPostTopo(TopoMap):
    def __init__(self, template: TemplateMap):
        super().__init__(TopoName.ANT_POST_GRADIENT, template)

    def load(self):
        if self.loaded:
            return
        self.template()
        voxels = cifti.read(self.file_full_path)[0][:, self.medial_wall_mask == 0]
        mask_no_wall = self.template.data.mask[self.medial_wall_mask == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str),
                          "coord_x": Series(dtype=float), "coord_y": Series(dtype=float),
                          "coord_z": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(self.template.data.regions, self.template.data.networks)):
            x, y, z = voxels[:, mask_no_wall == i + 1].mean(axis=1)
            topo.loc[i, :] = reg, net, x, y, z
        self._data = topo


class SchaeferTemplateMap(TemplateMap):
    _name_map = {"2007": TemplateName.SCHAEFER_200_7,
                 "20017": TemplateName.SCHAEFER_200_17}

    def __init__(self, space: Space, n_regions: int, n_networks: int):
        name = value_or_raise(self._name_map, f"{n_regions}{n_networks}",
                              f"SCHAEFER with {n_regions} regions and {n_networks} networks is not defined")
        super().__init__(name, space)
        self.n_regions = n_regions
        self.n_networks = n_networks

    @property
    def net_order(self):
        if self.n_networks == 7:
            return ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
        raise NotImplementedError(f"Network order for SCHAEFER with {self.n_networks} networks is not defined ")

    def net_hierarchy(self, name: HierarchyName):
        keys = name.keys
        if name == HierarchyName.PERIPHERY_CORE:
            return {keys[0]: ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn'], keys[1]: ['Limbic', 'Cont', 'Default']}
        raise NotImplementedError(f"{self.name} and {name} is not defined")

    def net_labels(self, break_space=True):
        if self.n_networks == 7:
            values = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Salience', 'Limbic', 'FPC', 'DMN']
        else:
            raise NotImplementedError(f"Network labels for SCHAEFER with {self.n_networks} networks is not defined ")
        return break_space_func(values) if break_space else values

    @property
    def net_colors(self):
        if self.n_networks == 7:
            return [
                [97 / 255, 46 / 255, 103 / 255],
                [119 / 255, 153 / 255, 191 / 255],
                [63 / 255, 152 / 255, 68 / 255],
                [210 / 255, 95 / 255, 199 / 255],
                [213 / 255, 227 / 255, 184 / 255],
                [238 / 255, 184 / 255, 57 / 255],
                [217 / 255, 112 / 255, 126 / 255],
            ]

        raise NotImplementedError(f"Network color for SCHAEFER with {self.n_networks} networks is not defined ")

    def load(self):
        if not self.loaded:
            mask, (lbl_axis, brain_axis) = cifti.read(self.file_full_path)
            mask = np.squeeze(mask)
            lbl_dict = lbl_axis.label.item()
            regions = np.asarray([lbl_dict[key][0] for key in list(lbl_dict.keys())])[1:]
            networks = [x.split("_")[2] for x in regions]
            unique_networks = np.unique(networks)
            self._data = TemplateData(mask, unique_networks, networks, regions, brain_axis)


class ColeTemplateMap(TemplateMap):
    def __init__(self, space: Space):
        super().__init__(TemplateName.COLE_360, space)

    @property
    def net_order(self):
        return ["Visual1", "Visual2", "Auditory", "Somatomotor", "PosteriorMultimodal", "VentralMultimodal",
                "OrbitoAffective", "DorsalAttention", "Language", "CinguloOpercular", "Frontoparietal", "Default"]

    def net_hierarchy(self, name: HierarchyName):
        keys = name.keys
        if name == HierarchyName.EXTENDED_PERIPHERY_CORE:
            return {keys[0]: ["Visual1", "Visual2", "Auditory", "Somatomotor"],
                    keys[1]: ["PosteriorMultimodal", "VentralMultimodal", "OrbitoAffective", "DorsalAttention",
                              "Language", "CinguloOpercular", "Frontoparietal", "Default"]}
        elif name == HierarchyName.RESTRICTED_PERIPHERY_CORE:
            return {keys[0]: ["Visual1", "Visual2", "Auditory", "Somatomotor"],
                    keys[1]: ["CinguloOpercular", "Frontoparietal", "Default"]}
        elif name == HierarchyName.LOWER_HIGHER_ORDER:
            return {keys[0]: ["Visual1", "Visual2", "Auditory", "Somatomotor", "PosteriorMultimodal",
                              "VentralMultimodal", "OrbitoAffective"],
                    keys[1]: ["DorsalAttention", "Language", "CinguloOpercular", "Frontoparietal", "Default"]}
        elif name == HierarchyName.UNI_TRANS_MODAL:
            return {keys[0]: ["Visual1", "Visual2", "Auditory", "Somatomotor"],
                    keys[1]: ["PosteriorMultimodal", "VentralMultimodal", "OrbitoAffective", "DorsalAttention",
                              "Language", "CinguloOpercular", "Frontoparietal", "Default"]}

        raise NotImplementedError(f"{self.name} and {name} is not defined")

    def net_labels(self, break_space=True):
        values = ['Visual1', 'Visual2', 'Auditory', 'Somatomotor', 'Posterior Multimodal', 'Ventral Multimodal',
                  'Orbito Affective', 'Dorsal Attention', 'Language', 'Cingulo Opercular', 'FPC', 'DMN']
        return break_space_func(values) if break_space else values

    @property
    def net_colors(self):
        return [
            [97 / 255, 46 / 255, 103 / 255],
            [114 / 255, 71 / 255, 83 / 255],
            [100 / 255, 125 / 255, 148 / 255],
            [119 / 255, 153 / 255, 191 / 255],
            [90 / 255, 130 / 255, 85 / 255],
            [31 / 255, 87 / 255, 39 / 255],
            [228 / 255, 177 / 255, 209 / 255],
            [63 / 255, 152 / 255, 68 / 255],
            [210 / 255, 95 / 255, 199 / 255],
            [213 / 255, 227 / 255, 184 / 255],
            [238 / 255, 184 / 255, 57 / 255],
            [217 / 255, 112 / 255, 126 / 255],
        ]

    def load(self):
        if not self.loaded:
            mask, (lbl_axis, brain_axis) = cifti.read(self.file_full_path)
            mask = np.squeeze(mask)
            lbl_dict = lbl_axis.label.item()
            regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
            networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
            unique_networks = np.unique(networks)
            self._data = TemplateData(mask, unique_networks, networks, regions, brain_axis)


class WangTemplateMap(TemplateMap):
    def __init__(self, space: Space):
        super().__init__(TemplateName.WANG, space)

    @property
    def net_order(self):
        return ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2", "TO1", "TO2",
                "LO1", "LO2", "V3A", "V3B", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1", "FEF"]

    def net_hierarchy(self, name: HierarchyName):
        raise NotImplementedError(f"{self.name} and {name} is not defined")

    def net_labels(self, break_space=True):
        values = self.net_order
        if break_space:
            return break_space_func(values)

    @property
    def net_colors(self):
        raise NotImplementedError(f"Colors are not defined for Wang template")

    def load(self):
        if not self.loaded:
            mask, (lbl_axis, brain_axis) = cifti.read(self.file_full_path)
            mask = np.squeeze(mask)
            lbl_dict = lbl_axis.label.item()
            regions = np.asarray([lbl_dict[x][0] for x in np.unique(mask)])[1:]
            networks = ["".join(x.split("_")[0].split("-")[:-1]) for x in regions]
            unique_networks = np.unique(networks)
            self._data = TemplateData(mask, unique_networks, networks, regions, brain_axis)
