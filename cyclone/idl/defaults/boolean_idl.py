from dataclasses import dataclass, field
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclonedds.idl import IdlStruct


@dataclass
class StateSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    state: int = field(metadata={"id": 1})
