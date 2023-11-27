import math
from typing import Optional, Sequence

import h5py
import numpy as np


class InferenceDataset:
    def __init__(
        self,
        fname: str,
        channels: Sequence[str],
        stride_size: int,
        shift_sizes: Optional[Sequence[int]] = None,
    ):
        """
        Simple Iterable dataset that chronologically loads
        `stride_size` windows of timeseries data.
        If `shift_sizes` is provided, the dataset will
        also yield windows that are shifted by the specified amounts.

        It is _strongly_ recommended that these files have been
        written using [chunked storage]
        (https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage).
        This has shown to produce increases in read-time speeds
        of over an order of magnitude.

        Args:
            fname:
                Paths to HDF5 file from which to load data.
            channels:
                Datasets to read from the indicated files, which
                will be stacked along dim 1 of the generated batches
                during iteration.
            stride_size:
                Size of the windows to read and yield at each step
            shift_sizes:
                List of shift sizes to apply to each channel. If `None`,
                no shifts will be applied.
        """

        self.fname = fname
        self.stride_size = stride_size
        self.channels = channels

        if shift_sizes is not None:
            if len(shift_sizes) != len(channels):
                raise ValueError("Shifts must be the same length as channels")
        self.shift_sizes = shift_sizes or [0] * len(channels)
        with h5py.File(fname, "r") as f:
            dset = f[channels[0]]
            self.size = len(dset) - self.max_shift

    def __len__(self):
        return math.ceil(self.size / self.stride_size)

    @property
    def max_shift(self):
        return max(self.shift_sizes)

    def __iter__(self):
        with h5py.File(self.fname, "r") as f:
            idx = 0
            while idx < self.size:
                data = []
                for channel, shift in zip(self.channels, self.shift_sizes):
                    start = idx + shift
                    stop = start + self.stride_size

                    # make sure that segments with shifts shorter
                    # than the max shift get their ends cut off
                    stop = min(self.size + shift, stop)
                    x = f[channel][start:stop]
                    data.append(x)

                yield np.stack(data)
                idx += self.stride_size