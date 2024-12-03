from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class JointConfigurationSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.UR5E_JOINT_CONFIGURATION}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    pose: array[float, 6] = field(metadata={"id": 1})
    velocity: array[float, 6] = field(metadata={"id": 2})
