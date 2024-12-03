from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, array

from cyclone.idl.defaults.rpc_idl import RPCIdl
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


class WorldConfigRPC(RPCIdl):
    @dataclass
    class Request(IdlStruct, typename=f"{CYCLONE_NAMESPACE.WORLD_CONFIG}_request.Msg"):
        timestamp: float = field(metadata={"id": 0})

    @dataclass
    class Response(IdlStruct, typename=f"{CYCLONE_NAMESPACE.WORLD_CONFIG}_response.Msg"):
        timestamp: float = field(metadata={"id": 0})

        cuboid: sequence[str] = field(metadata={"id": 1})
        cuboid_dims: sequence[array[float, 3]] = field(metadata={"id": 2})
        cuboid_pose: sequence[array[float, 7]] = field(metadata={"id": 3})
