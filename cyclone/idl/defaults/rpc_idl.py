from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct


class RPCIdl(ABC):
    @dataclass
    @abstractmethod
    class Request(IdlStruct, typename="RPCRequest.Msg"):
        timestamp: float = field(metadata={"id": 0})

    @dataclass
    @abstractmethod
    class Response(IdlStruct, typename="RPCResponse.Msg"):
        timestamp: float = field(metadata={"id": 0})
