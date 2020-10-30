import os
from enum import Enum


class AssetCategory(Enum):
    HCP1200 = "hcp1200"


def get_full_path(category: AssetCategory, name: str):
    if not name:
        raise ValueError("file name is None.")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), category.value, name)