from abc import abstractmethod, ABC
from neuro_helper.assets import manager
from enum import Enum
from pandas import DataFrame
import numpy as np
import cifti

__all__ = ["Space", "TopoName", "TemplateName", "HierarchyName", "TemplateData",
           "AbstractMap", "TemplateMap", "TopoMap", "file_names"]


class Space(Enum):
    K4 = "4k"
    K4_CORTEX = "4k_cortex"
    K32 = "32k"
    K32_CORTEX = "32k_cortex"
    K59 = "59k"
    K59_CORTEX = "59k_cortex"

    def __str__(self):
        return self.value

    @property
    def is_cortex(self):
        return "_cortex" in self.value

    @property
    def dropped_cortex(self):
        return Space(self.value.replace("_cortex", ""))


class TopoName(Enum):
    T1T2 = "t1t2"
    MARGULIES_GRADIENT = "gradient"
    ANT_POST_GRADIENT = "coord"
    MEDIAL_WALL = "medial_wall"

    def __str__(self):
        return self.value


class TemplateName(Enum):
    SCHAEFER_200_7 = "sh2007"
    SCHAEFER_200_17 = "sh20017"
    COLE_360 = "cole"
    WANG = "wang"

    def __str__(self):
        return self.value


class HierarchyName(Enum):
    LOWER_HIGHER_ORDER = "lh"
    PERIPHERY_CORE = "pc"
    EXTENDED_PERIPHERY_CORE = "pce"
    RESTRICTED_PERIPHERY_CORE = "pcr"
    UNI_TRANS_MODAL = "ut"

    def __str__(self):
        return self.value

    @property
    def keys(self):
        if self == HierarchyName.LOWER_HIGHER_ORDER:
            return "L", "H"
        elif self == HierarchyName.PERIPHERY_CORE:
            return "P", "C"
        elif self == HierarchyName.EXTENDED_PERIPHERY_CORE:
            return "P", "EC"
        elif self == HierarchyName.RESTRICTED_PERIPHERY_CORE:
            return "P", "RC"
        elif self == HierarchyName.UNI_TRANS_MODAL:
            return "U", "T"
        else:
            raise NotImplementedError(f"keys for {self} are not implemented")

    @property
    def labels(self):
        if self == HierarchyName.LOWER_HIGHER_ORDER:
            return "Lower-order", "Higher-order"
        elif self in [HierarchyName.PERIPHERY_CORE,
                      HierarchyName.EXTENDED_PERIPHERY_CORE,
                      HierarchyName.RESTRICTED_PERIPHERY_CORE]:
            return "Periphery", "Core"
        elif self == HierarchyName.UNI_TRANS_MODAL:
            return "Unimodal", "Transmodal"
        else:
            raise NotImplementedError(f"labels for {self} are not implemented")


class TemplateData:
    def __init__(self, mask, unique_networks, networks, regions, brain_axis):
        self.mask = mask
        self.unique_networks = unique_networks
        self.networks = networks
        self.regions = regions
        self.brain_axis = brain_axis

    def __call__(self, *args, **kwargs):
        return self.mask, self.unique_networks, self.networks, self.regions, self.brain_axis


class AbstractMap(ABC):
    asset_category = manager.AssetCategory.HCP1200
    space: Space
    _median_wall = None

    @property
    def file_full_path(self):
        return manager.get_full_path(self.asset_category, file_names[self.name][self.space])

    @property
    def medial_wall_mask(self):
        if self._median_wall is None:
            file_path = manager.get_full_path(self.asset_category, file_names[TopoName.MEDIAL_WALL][self.space])
            self._median_wall = cifti.read(file_path)[0].squeeze()
        return self._median_wall

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    def key(self):
        return f"{self.name}:{self.space}"

    @property
    def loaded(self):
        return self.data is not None

    def __init__(self, space: Space):
        self.space = space

    @abstractmethod
    def load(self):
        pass

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return f"{self.key} - Loaded: {self.loaded}"

    def __call__(self, *args, **kwargs):
        if not self.loaded:
            self.load()
        return self


class TemplateMap(AbstractMap, ABC):
    _name: TemplateName
    _data: TemplateData

    @property
    def data(self):
        return self._data

    def __init__(self, name: TemplateName, space: Space):
        super().__init__(space)
        self._name = name
        self._data = None

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def net_order(self):
        pass

    @abstractmethod
    def net_hierarchy(self, name: HierarchyName):
        pass

    @abstractmethod
    def net_labels(self, break_space=True):
        pass

    @property
    @abstractmethod
    def net_colors(self):
        pass

    def parcellate(self, voxels):
        mask, _, networks, regions, _ = self.data()

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


class TopoMap(AbstractMap, ABC):
    _name: TopoName
    template: TemplateMap
    _data: DataFrame

    @property
    def data(self):
        return self._data

    @property
    def template_loaded(self):
        return self.template.loaded

    @property
    def key(self):
        return f"{self.name}:{super().key}"

    @property
    def name(self):
        return self._name

    def __init__(self, name: TopoName, template: TemplateMap):
        super().__init__(template.space)
        self._name = name
        self._data = None
        self.template = template

    def __str__(self):
        return f"{super()} - Template: {self.template}"


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
