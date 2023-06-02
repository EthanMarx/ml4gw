# Torch implementations of signal processing functions. 
# Most functionality is lifted from gwpy: https://github.com/gwpy/gwpy/blob/main/gwpy/signal/filter_design.py

import torch
from ml4gw.types import ScalarTensor, BatchTimeSeriesTensor
from typing import Optional

def planck(N: int, nleft: int = 0, nright: int = 0):
    """Return a Planck taper window."""

    w = torch.ones(N, dtype=torch.float64)
    if nleft:
        w[0] *= 0
        zleft = torch.arange(1, nleft, dtype=torch.float64)
        zleft = nleft * (1./zleft + 1./(zleft-nleft))
        w[1:nleft] *= torch.special.expit(-zleft)
    if nright:
        w[N-1] *= 0
        zright = torch.arange(1, nright, dtype=torch.float64)
        zright = -nright * (1./(zright-nright) + 1./zright)
        w[N-nright:N-1] *= torch.special.expit(-zright)
    return w

def truncate_impulse(impulse: BatchTimeSeriesTensor, ntaps: ScalarTensor):
    trunc_start = (ntaps / 2).astype(torch.int8)
    size = impulse.size(-1)
    trunc_stop = size - trunc_start
    # TODO: implement other windowing functions
    window = torch.hann_window(ntaps)
    impulse[:, :, 0:trunc_start] *= window[trunc_start:ntaps]
    impulse[:, :, trunc_stop:impulse.size] *= window[0:trunc_start]
    impulse[:,:, trunc_start:trunc_stop] = 0
    return impulse


def truncate_transfer(transfer, ncorner: Optional[int] = None):
    out = transfer.clone()
    if out.ndim == 1:
        out = out[None, None, :]
    nsamp = out.size(-1)
    ncorner = ncorner if ncorner else 0
    out[:, :, 0:ncorner] = 0
    out[:, :, ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5)
    print(transfer[0][0][:10])
    return out

def fir_from_transfer(transfer, ntaps, ncorner=None):
    
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = torch.rfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps)
    # wrap around and normalise to construct the filter
    out = torch.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out