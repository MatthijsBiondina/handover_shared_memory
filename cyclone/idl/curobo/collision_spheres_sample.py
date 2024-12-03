from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, sequence

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class CuroboCollisionSpheresSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES}.Msg"):
    timestamp: float=field(metadata={"id": 0})

    positions: sequence[array[float, 3]] = field(metadata={"id": 1})