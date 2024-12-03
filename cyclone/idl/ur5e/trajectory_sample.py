from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, array

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class TrajectorySample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.CUROBO_TRAJECTORY}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    trajectory: sequence[array[float, 6]] = field(metadata={"id": 1})
