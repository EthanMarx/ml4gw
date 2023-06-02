import pytest
from gwpy.signal import filter_design as gwpy_filter_functions
from gwpy.signal.window import planck as gwpy_planck
import torch
from ml4gw.utils.signal import fir_from_transfer, truncate_transfer, truncate_impulse, planck
import numpy as np

@pytest.fixture(params=[1, 2, 3])
def n_channels(request):
    return request.param

@pytest.fixture(params=[2028])
def size(request):
    return request.param

@pytest.fixture(params=[2, 4, 8])
def batch_size(request):
    return request.param

@pytest.fixture
def timeseries(batch_size, n_channels, size):
    return torch.randn((batch_size, n_channels, size))

@pytest.fixture(params=[0, 2, 4])
def ncorner(request):
    return request.param

@pytest.fixture(params=[32, 64])
def ntaps(request):
    return request.param

@pytest.fixture(params=[0, 1, 2, 3])
def nleft(request):
    return request.param

@pytest.fixture(params=[0, 1, 2, 3])
def nright(request):
    return request.param

def test_planck_window(size, nleft, nright):
    w = planck(size, nleft=nleft, nright=nright)
    assert w.size(-1) == size
    assert np.allclose(w.numpy(), gwpy_planck(size, nleft, nright))


def test_truncate_transfer(timeseries, ncorner):
    transfer = truncate_transfer(timeseries, ncorner)
    
    timeseries = timeseries.numpy().reshape(-1, timeseries.shape[-1])
    transfer = transfer.numpy().reshape(-1, transfer.shape[-1])

    for tf, ts in zip(transfer, timeseries):
        gwpy_tf = gwpy_filter_functions.truncate_transfer(ts, ncorner)
        assert np.allclose(tf, gwpy_tf)


def test_truncate_impulse(timeseries, ncorner):
    transfer = truncate_impulse(timeseries, ncorner)
    
    timeseries = timeseries.numpy().reshape(-1, timeseries.shape[-1])
    transfer = transfer.numpy().reshape(-1, transfer.shape[-1])

    for tf, ts in zip(transfer, timeseries):
        gwpy_tf = gwpy_filter_functions.truncate_impulse(ts, ncorner)
        assert np.allclose(tf, gwpy_tf)

def test_fir_from_transfer(timeseries, ntaps):
    transfer = fir_from_transfer(timeseries, ntaps, ncorner=0)
   

    timeseries = timeseries.numpy().reshape(-1, timeseries.shape[-1])
    transfer = transfer.numpy().reshape(-1, transfer.shape[-1])
   
    for tf, ts in zip(transfer, timeseries):
        gwpy_tf = gwpy_filter_functions.fir_from_transfer(ts, ntaps, ncorner=0)
        assert np.allclose(tf, gwpy_tf)
    



