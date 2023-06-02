from torchtyping import TensorType
from typing import Union

WaveformTensor = TensorType["batch", "num_ifos", "time"]
PSDTensor = TensorType["num_ifos", "frequency"]
ScalarTensor = TensorType["batch"]
VectorGeometry = TensorType["batch", "space"]
TensorGeometry = TensorType["batch", "space", "space"]
NetworkVertices = TensorType["num_ifos", 3]
NetworkDetectorTensors = TensorType["num_ifos", 3, 3]
TimeSeriesTensor = Union[TensorType["time"], TensorType["channel", "time"]]
BatchTimeSeriesTensor = Union[
    TensorType["batch", "time"], TensorType["batch", "channel", "time"]
]