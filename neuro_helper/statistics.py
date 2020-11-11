import numpy as np
from scipy import signal, stats


def fir_filter(data, fs, order=None, min_cycles=4, max_freq_low=None, min_freq_high=None, pass_type="bp"):
    n_channel, n_sample = data.shape
    total_time = n_sample / fs

    if order is None:
        if n_sample > 800:
            order = 201
        elif 250 < n_sample < 350:
            order = 51
        else:
            raise Exception("Undefined number of points (%s) for filtering" % n_sample)

    freq_l = np.ceil(min_cycles / total_time * 1000) / 1000
    freq_h = np.floor(fs / 2 * 1000) / 1000

    if max_freq_low:
        freq_l = np.max([freq_l, max_freq_low])
    if min_freq_high:
        freq_h = np.min([freq_h, min_freq_high])

    if pass_type == "bp":
        pass_zero = False
        freq = [freq_l, freq_h]
    elif pass_type == "lp":
        pass_zero = True
        freq = freq_h
        freq_l = None
    elif pass_type == "hp":
        pass_zero = False
        freq = freq_l
        freq_h = None
    else:
        raise Exception("pass_type not defined")

    # noinspection PyTypeChecker
    return signal.filtfilt(signal.firwin(order, freq, window='hanning', pass_zero=pass_zero, fs=fs),
                           [1], data), freq_l, freq_h


def welch_psd(data, freq_l, fs):
    n_sample = data.shape[1]
    n_fft = 2 ** np.ceil(np.log2(n_sample))
    win_size = int(2 * 1.6 / freq_l * fs)

    if win_size > n_sample:
        return None, None

    return signal.welch(data, fs=fs, nfft=n_fft, detrend=None,
                        window="hanning", nperseg=win_size, noverlap=int(0.9 * win_size))


def percent_change(base_cond, new_cond):
    increase = new_cond - base_cond
    return increase / base_cond * 100


def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / (
                sum(aov['sum_sq']) + aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov


def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


def correlate_rows(data):
    n_items = data.shape[0]
    corr = np.zeros((n_items, n_items))
    for ri in range(n_items):
        for rj in range(ri, n_items):
            corr_val, _ = stats.pearsonr(data[ri, :], data[rj, :])
            corr[ri, rj] = corr[rj, ri] = corr_val

    return corr


def icc(y, icc_type='icc2'):
    """ Calculate intraclass correlation coefficient for data within
        Brain_Data class
    icc Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    Code from https://github.com/cosanlab/nltools/blob/master/nltools/data/brain_data.py
    Args:
        y: subjects x sessions
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        icc: (np.array) intraclass correlation coefficient
    """
    # n, k
    [n_subjects, n_scans] = y.shape

    # Degrees of Freedom
    dfc = n_scans - 1
    dfe = (n_subjects - 1) * dfc
    dfr = n_subjects - 1

    # Sum Square Total
    mean_y = np.mean(y)
    SST = ((y - mean_y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(n_scans), np.ones((n_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(n_subjects), (n_scans, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), y.flatten('F'))
    residuals = y.flatten('F') - predicted_y
    sse = (residuals ** 2).sum()

    mse = sse / dfe

    # Sum square column effect - between colums
    ssc = ((np.mean(y, 0) - mean_y) ** 2).sum() * n_subjects
    msc = ssc / dfc / n_subjects

    # Sum Square subject effect - between rows/subjects
    ssr = SST - ssc - sse
    msr = ssr / dfr

    if icc_type == 'icc1':
        # icc(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # icc = (msr - MSRW) / (msr + (k-1) * MSRW)
        raise NotImplementedError("This method isn't implemented yet.")
    elif icc_type == 'icc2':
        # icc(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        icc_value = (msr - mse) / (msr + (n_scans - 1) * mse + n_scans * (msc - mse) / n_subjects)
    elif icc_type == 'icc3':
        # icc(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        icc_value = (msr - mse) / (msr + (n_scans - 1) * mse)
    else:
        raise ValueError(f"{icc_type} is not defined.")

    return icc_value
