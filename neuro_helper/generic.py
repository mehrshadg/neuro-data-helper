import os
import numpy as np
import cifti
from pandas import DataFrame
from neuro_helper.template import get_template
from neuro_helper.entity import TemplateName, Space
import pandas_flavor as pf


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


def find_shared_subjects(find_files, prepare_file_content,
                         template_name: TemplateName, space: Space,
                         tasks, return_indices):
    all_ids = []
    subject_id_dict = {}
    for task in tasks:
        files = find_files(task=task, template_name=template_name, space=space)
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


def generate_long_data(find_files, prepare_file_content, template_name: TemplateName, space: Space):
    mask, unique_networks, networks, regions, _ = get_template(template_name, space)
    long_data = None
    for task in ["Rest", "StoryM", "Motort", "Wrkmem"]:
        files = find_files(task=task, template_name=template_name, space=space)
        for file in files:
            scan_id, subj_ids, metric = prepare_file_content(np.load(file, allow_pickle=True))
            n_subj, n_regions = metric.shape
            task_scan = "%s-%s" % (task, scan_id)
            cols = np.c_[
                np.repeat(task, n_subj * n_regions),
                np.repeat(scan_id, n_subj * n_regions),
                np.repeat(task_scan, n_subj * n_regions),
                np.repeat(subj_ids, n_regions),
                np.tile(networks, n_subj),
                np.tile(regions, n_subj),
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
def build_single_topo_map(df: DataFrame, template_name: TemplateName, space: Space):
    """
    :param df: must only contains two columns: regions and values
    :param template_name:
    :param space:
    :return:
    """
    mask, _, networks, regions, brain_axis = get_template(template_name, space)
    topo = np.zeros((1, mask.size))
    values = df.values
    for reg, pc in values:
        reg_index = np.argmax(regions == reg) + 1
        if reg_index == 0:
            print("0 reg_index in %s" % reg)
        topo[0, np.argwhere(mask == reg_index)] = pc
    return topo, brain_axis


def combine_topo_map(topo_maps):
    topos, brains = list(zip(*topo_maps))
    return np.concatenate(topos, axis=0), brains[0], cifti.Series(0, 1, len(topo_maps))


