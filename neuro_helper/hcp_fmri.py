import help
import glob
import os
import cifti
import numpy as np

from neuro_helper.entity import Space


def get_all_raw_files(task_name, mode, id, space: Space):
    if space == Space.K32:
        pattern = f"/data/hcp/3T/#SUBJ#/MNINonLinear/Results/[tr]fMRI_{task_name}#SCAN#_[LR][LR]/*_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    elif space == Space.K64:
        pattern = f"/data/hcp/7T/#SUBJ#/MNINonLinear/Results/[tr]fMRI_{task_name}#SCAN#_7T_*/*_7T_*_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries.nii"
    else:
        raise ValueError(f"{space} doesn't have fMRI HCP raw files")

    if mode == "by_subj":
        pattern = pattern.replace("#SUBJ#", id).replace("#SCAN#", "*")
    elif mode == "by_scan":
        pattern = pattern.replace("#SUBJ#", "*").replace("#SCAN#", id)
    else:
        raise Exception("%s mode is not defined" % mode)

    files = glob.glob(pattern)
    files.sort()

    dictionary = {}
    for file in files:
        path_parts = file.split(os.sep)
        subj_id = path_parts[-5]
        scan_id = path_parts[-2].split("_")[1].replace(task_name, "")

        key = scan_id if mode == "by_scan" else "SUBJ"
        if key not in dictionary:
            dictionary[key] = {}
        if subj_id not in dictionary[key]:
            dictionary[key][subj_id] = []

        dictionary[key][subj_id].append((scan_id, file))

    return dictionary


def load_raw_file(file: str, space: Space):
    data = cifti.read(file)[0].T
    if space == Space.K64:
        fs = 1.0
    elif space == Space.K32:
        fs = 1000 / 720
    else:
        raise ValueError(f"{space} is not defined in loading a single raw file")
    return data, fs


def load_raw_files(input, space: Space, merge_func, **kwargs):
    if type(input) == str:
        data, fs = load_raw_file(input, space)
        return data, fs, input
    elif type(input) == list and len(input) == 1:
        data, fs = load_raw_file(input[0][1], space)
        return data, fs, input[0][1]
    elif type(input) == list:
        _, files = list(zip(*input))
        return merge_func(files, space, **kwargs)
    else:
        raise ValueError("input type is not valid")


def concat_scans(files, space: Space, **kwargs):
    cut_ratio = kwargs.get("cut_ratio")
    cut_sample = kwargs.get("cut_sample")
    output = []
    shared_fs = None

    for file in files:
        data, fs = load_raw_file(file, space)
        if shared_fs is None:
            shared_fs = fs
        elif not fs == shared_fs:
            raise Exception("Incompatible FS, cannot merge scans in %s" % files)

        if cut_ratio is not None:
            data_cut_sample = int(data.shape[1] * cut_ratio)
        elif cut_sample is not None:
            data_cut_sample = cut_sample
        else:
            data_cut_sample = data.shape[1]

        if data_cut_sample > data.shape[1]:
            raise Exception("Large cut sample in merging %s" % file)
        output.append(data[:, :data_cut_sample])

    return np.column_stack(output), shared_fs, "CONCAT " + ", ".join(files)


def average_scans(files, space: Space, **kwargs):
    output = []
    shared_fs = None

    min_length = np.Inf
    for file in files:
        data, fs = load_raw_file(file, space)
        if shared_fs is None:
            shared_fs = fs
        elif not fs == shared_fs:
            print(f"Shared Fs: {shared_fs} and Fs: {fs}")
            raise Exception(f"Incompatible FS, cannot merge scans in {', '.join(files)}")

        min_length = min(min_length, data.shape[1])
        output.append(data)

    output = np.asarray([x[:, :min_length] for x in output])
    return output.mean(axis=0), shared_fs, "AVERAGE " + ", ".join(files)



