# Torch implementations of signal processing functions. 
# Most functionality is lifted from gwpy: https://github.com/gwpy/gwpy/blob/main/gwpy/signal/filter_design.py

import torch


def truncate_impulse(impulse: BatchTimeSeriesTensor, ntaps: ScalarTensor):
    """Smoothly truncate a time domain impulse response
    Parameters
    ----------
    impulse : `numpy.ndarray`
        the impulse response to start from

    ntaps : `int`
        number of taps in the final filter

    Returns
    -------
    out : `numpy.ndarray`
        the smoothly truncated impulse response
    """
    out = impulse.copy()
    trunc_start = int(ntaps / 2)
    trunc_stop = out.size - trunc_start
    # TODO: implement other windowing functions
    window = torch.hann_window()
    out[0:trunc_start] *= window[trunc_start:ntaps]
    out[trunc_stop:out.size] *= window[0:trunc_start]
    out[trunc_start:trunc_stop] = 0
    return out


def truncate_transfer():
    pass


def fir_from_transfer(transfer, ntaps, window='hann', ncorner=None):
    
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = torch.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = torch.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out