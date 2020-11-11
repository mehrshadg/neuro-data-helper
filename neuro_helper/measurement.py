import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from statsmodels.tsa import stattools
from neuro_helper.statistics import fir_filter
from scipy.signal import hilbert
import itertools


def _calc_parallel(func, data, n_job, threading, **kwargs):
    if n_job is None:
        n_job = data.shape[0]
    elif n_job == 0:
        raise ValueError("n_job cannot be zero")
    elif n_job > data.shape[0]:
        raise ValueError(f"n_job cannot be larger than the number of rows in data")
    backend = 'threading' if threading else None
    return np.asarray(Parallel(n_jobs=n_job, backend=backend)(delayed(func)(ts, **kwargs) for ts in data))


def lempel_ziv_complexity(sequence):
    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind: ind + inc]
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


def calc_acf(data, n_lag=None, fast=True, n_job=None, threading=False):
    return _calc_parallel(calc_a_acf, data, n_job, threading, n_lag=n_lag, fast=fast)


def calc_a_acw(ts, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return np.argmax(acf < 0.5)


def calc_acw(data, n_lag=None, fast=True, is_acf=False, n_job=None, threading=False):
    return _calc_parallel(calc_a_acw, data, n_job, threading, n_lag=n_lag, fast=fast, is_acf=is_acf)


def calc_a_acz(ts, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return np.argmax(acf <= 0)


def calc_acz(data, n_lag=None, fast=True, is_acf=False, n_job=None, threading=False):
    return _calc_parallel(calc_a_acz, data, n_job, threading, n_lag=n_lag, fast=fast, is_acf=is_acf)


def calc_a_acmi(ts, which, n_lag=None, fast=True, is_acf=False):
    acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
    return signal.argrelextrema(acf, np.less)[0][which - 1]


def calc_acmi(data, which, n_lag=None, fast=True, is_acf=False, n_job=None, threading=False):
    return _calc_parallel(calc_a_acmi, data, n_job, threading, which=which, n_lag=n_lag, fast=fast, is_acf=is_acf)


def calc_lzc_norm_factor(ts):
    """
    The default way of calculating LZC normalization factor for a time series
    :param ts: a time series
    :return: normalization factor
    """
    return len(ts) / np.log2(len(ts))


def calc_a_lzc(ts, norm_factor=None):
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


def calc_lzc(data, norm_factor=None, n_job=None, threading=False):
    return _calc_parallel(calc_a_lzc, data, n_job, threading, norm_factor=norm_factor)


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


def calc_a_ple(a_psd, freq, is_log=False):
    if not is_log:
        freq = np.log(freq)
        a_psd = np.log(a_psd)
    return np.polyfit(freq, a_psd, deg=1)


def calc_ple(psd, freq, is_log=False, n_job=None, threading=False):
    return _calc_parallel(calc_a_ple, psd, n_job, threading, freq=freq, is_log=is_log)


def calc_peak(data, fs, low, high, apply_median_filter, n_job=None):
    data, freq_l, freq_h = fir_filter(data, fs, max_freq_low=low, min_freq_high=high, pass_type="bp")
    n_wave, n_sample = data.shape
    hilb_data = hilbert(data, n_sample, axis=0)
    phase = np.angle(hilb_data)
    instant_freq = fs * np.diff(np.unwrap(phase)) / (2 * np.pi)

    if not apply_median_filter:
        return instant_freq

    # apply median filter to attenuate large "blips" in instantaneous frequency
    n_window = 10
    win_lengths = np.round(np.linspace(10, 400, n_window) * fs / (2 * 1000)).astype(int)

    def win_calc(wl, si):
        start, stop = max(si - wl, 0), min(si + wl, n_sample)
        return np.median(np.sort(instant_freq[:, start: stop], axis=1), axis=1)

    n_job = n_job if n_job else 10
    phased_med = np.zeros((n_window, n_sample, n_wave))
    for wi, wl in enumerate(win_lengths):
        phased_med[wi, :, :] = np.asarray(Parallel(n_jobs=n_job)(delayed(win_calc)(wl, si) for si in range(n_sample)))
        # for si in range(n_sample):
        #     start, stop = max(si - win_lengths[wi], 0), min(si + win_lengths[wi], n_sample)
        #     phased_med[wi, si, :] = np.median(np.sort(instant_freq[:, start: stop], axis=1), axis=1)

    sliding_instant_freq = np.mean(phased_med, axis=0).T
    return sliding_instant_freq


def calc_ratio_occurrence(data_a, data_b, decimals=0, axis=None):
    precision = 1 / 10 ** decimals
    data_r = np.round(data_a / data_b, decimals)
    possible_ratios = np.arange(data_r.min(), data_r.max() + precision, precision)
    normalization_coeff = data_r.size if axis is None else data_r.shape[axis]
    return np.asarray([
        np.sum(data_r == np.round(ratio, decimals), axis=axis) / normalization_coeff * 100
        for ri, ratio in enumerate(possible_ratios)
    ])
