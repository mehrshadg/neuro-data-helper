import numpy as np
from scipy import signal
from statsmodels.tsa import stattools


def fir_filter_data(data, fs, order=None, min_cycles=4, max_freq_low=None, min_freq_high=None, pass_type="bp"):
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


def icc(Y, icc_type='icc2'):
    """ Calculate intraclass correlation coefficient for data within
        Brain_Data class
    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    Code from https://github.com/cosanlab/nltools/blob/master/nltools/data/brain_data.py
    Args:
        Y: subjects x sessions
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    """
    # n, k
    [n_subjects, n_scans] = Y.shape

    # Degrees of Freedom
    dfc = n_scans - 1
    dfe = (n_subjects - 1) * dfc
    dfr = n_subjects - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(n_scans), np.ones((n_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(n_subjects), (n_scans, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n_subjects
    MSC = SSC / dfc / n_subjects

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'icc2':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (n_scans - 1) * MSE + n_scans * (MSC - MSE) / n_subjects)

    elif icc_type == 'icc3':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (n_scans - 1) * MSE)

    return ICC
