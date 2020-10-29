import itertools
import pandas as pd
import numpy as np
from matplotlib.cbook import boxplot_stats
import pandas_flavor as pf
from neuro_helper.entity import Space, TemplateName
from neuro_helper.statistics import percent_change, icc


@pf.register_dataframe_method
def add_net_meta(df, labels):
    meta = pd.Series(index=df.index, name="net_meta")
    for label, nets in labels.items():
        meta.loc[df.network.isin(nets)] = label
    df["net_meta"] = meta
    return df


@pf.register_dataframe_method
def convert_column(df, **col_dict):
    new_df = df.copy()
    for col_name, func in col_dict.items():
        new_df[col_name] = func(new_df[col_name])
    return new_df


@pf.register_dataframe_method
def and_filter(df, drop_single=True, **kwargs):
    filt = True
    keys = []
    for key, value in kwargs.items():
        negate = False
        if key.startswith("NOT"):
            negate = True
            key = key.replace("NOT", "")

        keys.append(key)
        if type(value) in [list, tuple, np.ndarray]:
            this_filt = df[key].isin(value)
        else:
            this_filt = df[key] == value

        filt &= ~this_filt if negate else this_filt

    new_df = df[filt]
    if drop_single:
        return new_df.drop([c for c in keys if len(new_df[c].unique()) <= 1], 1)
    else:
        return new_df


@pf.register_dataframe_method
def get_outlier_bounds(df, of):
    if isinstance(of, str):
        of = [of, ]

    out = []
    for col in of:
        stat = boxplot_stats(df[col])[0]
        out.append((stat["whislo"], stat["whishi"]))

    return out[0] if len(out) == 1 else out


@pf.register_dataframe_method
def avg_over_net(df):
    return df.groupby(list(df.columns.drop(["region", "metric"]))).mean().reset_index()


@pf.register_dataframe_method
def normalize(x, columns, new_min=0, new_max=1):
    if isinstance(columns, str):
        columns = [columns, ]
    df = x.copy()
    for on in columns:
        df[on] = normalize_series(df[on], new_min, new_max)
    return df


@pf.register_dataframe_method
def add_topo(df, template_name: TemplateName, space: Space, *args):
    from neuro_helper.template import get_topo_dataframe
    new_df = df
    has_net = "network" in df.columns
    for arg in args:
        topo = get_topo_dataframe(arg, template_name, space)
        if has_net:
            new_df = pd.merge(new_df, topo, on=["region", "network"])
        else:
            topo = topo.drop("network", 1)
            new_df = pd.merge(new_df, topo, on=["region"])

    return new_df


@pf.register_dataframe_method
def add_median_lh(x, calc_med_col, values=("L", "H")):
    med = x[calc_med_col].median()
    return add_split_label(x, calc_med_col, calc_med_col, (values, med))


@pf.register_dataframe_method
def add_split_label(df, on, based, criteria):
    x = df.copy()
    if callable(criteria):
        labels, borders = criteria(x[based])
    else:
        labels, borders = criteria

    if np.isscalar(borders):
        borders = [borders, ]

    if len(labels) != len(borders) + 1:
        raise ValueError("labels should be one more than borders")

    new_col_name = f"{on}_split"
    on_splitted = pd.Series(index=x.index, name=new_col_name, data=pd.NA)

    borders.append(borders[-1])
    for index, (label, border) in enumerate(zip(labels, borders)):
        if index == 0:
            filt = x[based] < border
        elif index == len(labels) - 1:
            filt = x[based] >= border
        else:
            filt = (x[based] < border) & (x[based] >= borders[index - 1])
        on_splitted.loc[filt] = label

    if on_splitted.isna().any():
        raise ValueError(f"criteria does not cover the whole {on} bound")

    x[new_col_name] = on_splitted
    return x


@pf.register_dataframe_method
def remove_outliers(x, of):
    stat = boxplot_stats(x[of])[0]
    low, high = stat["whislo"], stat["whishi"]
    return x.loc[(x[of] > low) & (x[of] < high)]


@pf.register_dataframe_method
def calc_paired_diff(x, diff_func=lambda left, right: abs(left - right), repeat=True):
    """
    calculates the 2-by-2 difference on one single column.
    :param x: a dataframe with only two columns.
    First column is the label for the second column. The second one is the metric
    :param diff_func: the difference function. default is L1 norm
    :param repeat: if True return all combinations with repeteation (product), otherwise only unique combinations
    :return: a dataframe with 3 columns. Left items, Right items and the difference between left and right
    """
    diff = pd.DataFrame(columns=("left", "right", "difference"))
    iterator = itertools.product(range(len(x)), repeat=2) if repeat else itertools.combinations(range(len(x)), 2)
    index = 0
    for li, ri in iterator:
        if li == ri:
            continue
        diff.loc[index, :] = x.iloc[li, 0], x.iloc[ri, 0], diff_func(x.iloc[li, 1], x.iloc[ri, 1])
        index += 1

    diff.difference = diff.difference.astype(np.float)
    return diff


@pf.register_dataframe_method
def calc_pchange(x):
    a = x[x.task == "Rest"]
    if not a.shape[0] == 1:
        raise Exception("Rest does not exist for %s" % x)
    rest_val = a.metric.item()
    output = []
    for task in x.task:
        if task == "Rest":
            continue
        output.append([task, percent_change(rest_val, x[x.task == task].metric.item())])

    df = pd.DataFrame(output, columns=["task", "pchange"])
    df.pchange = df.pchange.astype(float)
    return df


@pf.register_dataframe_method
def calc_icc(x):
    return pd.Series(
        {"icc": icc(x.drop("region", 1).pivot(index='subject', columns='scan', values='metric').values)})


@pf.register_series_method
def normalize_series(x, new_min=0, new_max=1):
    old_min = x.min()
    old_max = x.max()
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((x - old_min) * new_range) / old_range) + new_min


def concat_dfs(by, on, new_col="cat", **dfs):
    df = pd.DataFrame(columns=on + [by, new_col])
    for label, x in dfs.items():
        temp = pd.DataFrame(data=x, columns=on + [by, new_col])
        temp[new_col] = label
        df = df.append(temp, ignore_index=True, sort=False)

    return df.reset_index(drop=True)
