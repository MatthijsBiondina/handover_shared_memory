from dataclasses import dataclass, field
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclonedds.idl import IdlStruct


@dataclass
class FloatSample(IdlStruct, typename=f"Float.Msg"):
    timestamp: float = field(metadata={"id": 0})
    value: float = field(metadata={"id": 1})
