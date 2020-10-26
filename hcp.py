import re
from scipy.io import loadmat
import glob
import numpy as np


def get_all_raw_files(mask_lbl, task_name, scan_id):
    if task_name == "Rest":
        root_dir = "/group/northoff/share/mg/hcp/rest"
        task_name = "Restin"
    else:
        root_dir = "/group/northoff/share/mg/hcp/task"

    files = glob.glob("%s/*_MEG_%s-%s_*[tr]megpreproc_rois-%s.mat" % (root_dir, scan_id, task_name, mask_lbl))
    files.sort()

    scans = {}
    for file in files:
        found = re.findall('_[0-9]+-', file)
        if not len(found) == 1:
            print("WARNING: Cannot find unique scan id in %s" % file)
            continue
        scan_id = int(found[0][1:-1])

        found = re.findall('[0-9]+_MEG', file)
        if not len(found) == 1:
            print("WARNING: Cannot find unique subject id in %s" % file)
            continue
        subj_id = found[0].replace("_MEG", "")

        if scan_id not in scans:
            scans[scan_id] = []
        scans[scan_id].append((subj_id, file))

    return scans


def load_raw_file(file):
    raw_data = loadmat(file)["data"]
    fs = raw_data["fsample"].item().item()
    data = np.concatenate(raw_data["trial"].item()[0], axis=1)

    return data, fs

