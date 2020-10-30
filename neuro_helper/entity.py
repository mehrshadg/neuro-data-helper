from enum import Enum


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
