from dataclasses import field, dataclass
from cyclonedds.idl import IdlStruct
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class CoordinateSample(IdlStruct, typename="Coordinate.Msg"):
    timestamp: float = field(metadata={"id": 0})
    x: float = field(metadata={"id": 1})
    y: float = field(metadata={"id": 2})
    z: float = field(metadata={"id": 3})
