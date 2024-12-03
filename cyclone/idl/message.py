from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct


@dataclass
class Message(IdlStruct, typename="Message.Msg"):
    id: int = field(metadata={"id": 0})
    content: str = field(metadata={"id": 1})
