import numpy as np
from scipy import signal
from statsmodels.tsa import stattools


def lempel_ziv_complexity(sequence):
    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)


def calc_a_acf(ts, n_lag=None, fast=True):
    if not n_lag:
        n_lag = len(ts)
    return stattools.acf(ts, nlags=n_lag, qstat=False, alpha=None, fft=fast)


def calc_a_acw(ts, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return np.argmax(acf < 0.5)


def calc_a_acz(ts, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return np.argmax(acf <= 0)


def calc_a_acmi(ts, which, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return signal.argrelextrema(acf, np.less)[0][which - 1]


def calc_lzc_norm_factor(ts):
    """
    The default way of calculating LZC normalization factor for a time series
    :param ts: a time series
    :return: normalization factor
    """
    return len(ts) / np.log2(len(ts))


def calc_lzc(ts, norm_factor=None):
    """
    Calculates lempel-ziv complexity of a single time series.
    :param ts: a time-series: nx1
    :param norm_factor: the normalization factor. If none, the default value will be calculated: n / ln(n)
    :return: the lempel-ziv complexity
    """
    if not norm_factor:
        norm_factor = calc_lzc_norm_factor(ts)
    bin_ts = np.char.mod('%i', ts >= np.median(ts))
    return lempel_ziv_complexity("".join(bin_ts)) / norm_factor


def calc_mf(freq, psd):
    """
    Calculates median frequency of different power spectral densities.
    n is number of PSDs and m in number of frequencies.
    :param freq: the mx1 numpy array of frequencies.
    :param psd: the nxm numpy matrix of power.
    :return: nx1 array of median frequencies.
    """
    cum_sum = np.cumsum(psd, axis=1)
    return freq[np.argmax(cum_sum >= cum_sum[:, -1].reshape(-1, 1) / 2, axis=1)]
