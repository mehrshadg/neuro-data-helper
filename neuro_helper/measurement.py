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
    return 2 * np.argmax(acf < 0.5) - 1


def calc_a_acz(ts, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return np.argmax(acf <= 0)


def calc_a_acmi(ts, which, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return signal.argrelextrema(acf, np.less)[0][which - 1]


def calc_lzc(ts, norm_factor):
    bin_ts = np.char.mod('%i', ts >= np.median(ts))
    return lempel_ziv_complexity("".join(bin_ts)) / norm_factor


def calc_mf(freq, psd):
    cum_sum = np.cumsum(psd, axis=1)
    return freq[np.argmax(cum_sum >= cum_sum[:, -1].reshape(-1, 1) / 2, axis=1)]