from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct


@dataclass
class BufferNrSample(IdlStruct, typename="BufferNr.Msg"):
    timestamp: float = field(metadata={"id": 0})

    nr: int = field(metadata={"id": 1})
