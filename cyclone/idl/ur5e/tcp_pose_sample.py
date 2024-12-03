from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class TCPPoseSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.UR5E_TCP_POSE}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    pose: array[array[float, 4], 4] = field(metadata={"id": 1})
    velocity: array[array[float, 4], 4] = field(metadata={"id": 2})
