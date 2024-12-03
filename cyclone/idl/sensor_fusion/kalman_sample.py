from dataclasses import dataclass, field
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, sequence

@dataclass
class KalmanSample(IdlStruct, typename="KalmanSample.Msg"):
    timestamp: float = field(metadata={"id": 0})

    mean: sequence[array[float, 3]] = field(metadata={"id": 1})
    covariances: sequence[array[array[float, 3], 3]] = field(metadata={"id": 2})