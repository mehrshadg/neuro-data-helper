import re
from scipy.io import loadmat
import numpy as np
from neuro_helper.entity import TemplateName
from neuro_helper.storage import LocalStorage, StorageFile, ANYTHING, RCloneStorage


__all__ = ["MEGLocalStorage", "MEGRCloneStorage", "load_raw_file", "task_order"]


class MEGLocalStorage(LocalStorage):
    def __init__(self, root: str, tpt_name: TemplateName, task_name: str, scan_id: str):
        if task_name == "Rest":
            task_name = "Restin"
        super().__init__(root, [ANYTHING, "_MEG_", "scan_id", "-", "task_name",
                                ANYTHING, "megpreproc_rois-", "tpt_name", ".mat"],
                         **{"scan_id": scan_id, "task_name": task_name, "tpt_name": tpt_name})

    def get_all_by_scan(self):
        files = self.get_all()

        scans = {}
        for file in files:
            file_str = file.name
            found = re.findall('_[0-9]+-', file_str)
            if not len(found) == 1:
                print(f"WARNING: Cannot find unique scan id in {file_str}")
                continue
            scan_id = found[0][1:-1]

            found = re.findall('[0-9]+_MEG', file_str)
            if not len(found) == 1:
                print(f"WARNING: Cannot find unique subject id in {file_str}")
                continue
            subj_id = found[0].replace("_MEG", "")

            if scan_id not in scans:
                scans[scan_id] = []
            scans[scan_id].append((subj_id, file))

        return scans


class MEGRCloneStorage(RCloneStorage):
    def __init__(self, remote: str, root: str, task_name: str, scan_id: str):
        if task_name == "Rest":
            task_name = "Restin"
        super().__init__(
            remote, root, [ANYTHING, "_MEG_", "scan_id", "-", "task_name", ANYTHING, "megpreproc_voxels.mat"],
            **{"scan_id": scan_id, "task_name": task_name})

    def get_all_by_scan(self):
        files = self.get_all()

        scans = {}
        for file in files:
            file_str = file.name
            found = re.findall('_[0-9]+-', file_str)
            if not len(found) == 1:
                print(f"WARNING: Cannot find unique scan id in {file_str}")
                continue
            scan_id = found[0][1:-1]

            found = re.findall('[0-9]+_MEG', file_str)
            if not len(found) == 1:
                print(f"WARNING: Cannot find unique subject id in {file_str}")
                continue
            subj_id = found[0].replace("_MEG", "")

            if scan_id not in scans:
                scans[scan_id] = []
            scans[scan_id].append((subj_id, file))

        return scans


def load_raw_file(file: StorageFile):
    raw_data = loadmat(file.loadable_path)["data"]
    fs = raw_data["fsample"].item().item()
    data = np.concatenate(raw_data["trial"].item()[0], axis=1)

    return data, fs


def task_order(with_rest=True):
    if with_rest:
        out = ["Rest"]
    else:
        out = []
    return out + ["StoryM", "Motort", "Wrkmem"]
