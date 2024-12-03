from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, uint16, uint8

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class D405FrameSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.D405_FRAME}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    color: array[array[array[uint8, 3], 848], 480] = field(metadata={"id": 1})
    depth: array[array[uint16, 848], 480] = field(metadata={"id": 2})
    cam_pose: array[array[float, 4], 4] = field(metadata={"id": 3})
