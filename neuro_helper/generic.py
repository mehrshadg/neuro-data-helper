import os
import numpy as np
import cifti
from pandas import DataFrame
from neuro_helper.abstract.map import TemplateMap
import pandas_flavor as pf


__all__ = ["value_or_raise", "break_space", "out_of", "find_shared_subjects",
           "generate_long_data", "build_single_topo_map", "combine_topo_map"]


def value_or_raise(data: dict, key: str, msg: str):
    value = data.get(key)
    if value is None:
        raise KeyError(msg)
    return value


def break_space(values):
    return [x.replace(" ", "\n") for x in values]


def out_of(name, dir_from_last_part=True):
    file = "outputs"

    name_none, ext = os.path.splitext(name)

    for checker in [".dlabel", ".dtseries", ".shape"]:
        if checker in name_none:
            name_none = name_none.replace(checker, "")
            ext = checker + ext

    if ext in [".png", ".jpg", ".jpeg"]:
        file += "/figures"

    name_splitted = name_none.split(".")
    middle_dirs = os.sep.join(name_splitted[:-1])
    last_dir = name_splitted[-1].split("-")[0] if dir_from_last_part else ""

    file = os.path.join(file, middle_dirs, last_dir, name)

    file_dir = os.path.dirname(file)
    if "*" not in file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file


def find_shared_subjects(find_files, prepare_file_content, template: TemplateMap, tasks, return_indices):
    all_ids = []
    subject_id_dict = {}
    for task in tasks:
        files = find_files(task=task, template=template)
        for file in files:
            scan_id, subj_ids = prepare_file_content(np.load(file, allow_pickle=True))
            all_ids += subj_ids
            subject_id_dict["%s-%s" % (task, scan_id)] = np.asarray(subj_ids)

    all_ids.sort()
    all_ids, count = np.unique(all_ids, return_counts=True)
    shared_ids = all_ids[count == len(subject_id_dict.keys())]
    keys = list(subject_id_dict.keys())
    keys.sort()
    shared_indices = np.asarray([np.argwhere(np.isin(subject_id_dict[key], shared_ids)).flatten() for key in keys])
    if return_indices:
        return shared_ids, keys, shared_indices
    else:
        return shared_ids


def generate_long_data(find_files, prepare_file_content, template: TemplateMap, tasks, show_warning=True):
    t = template().data
    long_data = None
    for task in tasks:
        files = find_files(task=task, template=template)
        for file in files:
            scan_id, subj_ids, metric = prepare_file_content(np.load(file, allow_pickle=True))
            n_subj, n_regions = metric.shape
            if len(t.regions) < n_regions:
                if show_warning:
                    print(f"Data {file}")
                    print(f"number of regions in the template ({len(t.regions)}) "
                          f"is not the same as the number of regions in the data ({n_regions}). Trimming.")
                n_regions = len(t.regions)
                metric = metric[:, :n_regions]
            task_scan = "%s-%s" % (task, scan_id)
            cols = np.c_[
                np.repeat(task, n_subj * n_regions),
                np.repeat(scan_id, n_subj * n_regions),
                np.repeat(task_scan, n_subj * n_regions),
                np.repeat(subj_ids, n_regions),
                np.tile(t.networks, n_subj),
                np.tile(t.regions, n_subj),
                metric.flatten()
            ]
            if long_data is None:
                long_data = cols
            else:
                long_data = np.r_[long_data, cols]
    df = DataFrame(long_data, columns=["task", "scan", "task_scan", "subject", "network", "region", "metric"])
    df.metric = df.metric.astype(np.float)
    return df


@pf.register_dataframe_method
def build_single_topo_map(df: DataFrame, template: TemplateMap):
    """
    :param df: must only contains two columns: regions and values
    :param template:
    :return:
    """
    t = template().data
    topo = np.zeros((1, t.mask.size))
    values = df.values
    for reg, pc in values:
        reg_index = np.argmax(t.regions == reg) + 1
        if reg_index == 0:
            print("0 reg_index in %s" % reg)
        topo[0, np.argwhere(t.mask == reg_index)] = pc
    return topo, t.brain_axis


def combine_topo_map(topo_maps):
    topos, brains = list(zip(*topo_maps))
    return np.concatenate(topos, axis=0), brains[0], cifti.Series(0, 1, len(topo_maps))
