from dataclasses import dataclass, field
from typing import Optional
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE


@dataclass
class HandSample(IdlStruct, typename=f"{CYCLONE_NAMESPACE.MEDIAPIPE_HAND}.Msg"):
    timestamp: float = field(metadata={"id": 0})

    thumb_mcp: Optional[array[float, 5]] = field(default=None, metadata={"id": 1})
    thumb_ip: Optional[array[float, 5]] = field(default=None, metadata={"id": 2})
    thumb_tip: Optional[array[float, 5]] = field(default=None, metadata={"id": 3})

    index_finger_mcp: Optional[array[float, 5]] = field(default=None, metadata={"id": 4})
    index_finger_pip: Optional[array[float, 5]] = field(default=None, metadata={"id": 5})
    index_finger_dip: Optional[array[float, 5]] = field(default=None, metadata={"id": 6})
    index_finger_tip: Optional[array[float, 5]] = field(default=None, metadata={"id": 7})

    middle_finger_mcp: Optional[array[float, 5]] = field(default=None, metadata={"id": 8})
    middle_finger_pip: Optional[array[float, 5]] = field(default=None, metadata={"id": 9})
    middle_finger_dip: Optional[array[float, 5]] = field(default=None, metadata={"id": 10})
    middle_finger_tip: Optional[array[float, 5]] = field(default=None, metadata={"id": 11})

    ring_finger_mcp: Optional[array[float, 5]] = field(default=None, metadata={"id": 12})
    ring_finger_pip: Optional[array[float, 5]] = field(default=None, metadata={"id": 13})
    ring_finger_dip: Optional[array[float, 5]] = field(default=None, metadata={"id": 14})
    ring_finger_tip: Optional[array[float, 5]] = field(default=None, metadata={"id": 15})

    pinky_mcp: Optional[array[float, 5]] = field(default=None, metadata={"id": 16})
    pinky_pip: Optional[array[float, 5]] = field(default=None, metadata={"id": 17})
    pinky_dip: Optional[array[float, 5]] = field(default=None, metadata={"id": 18})
    pinky_tip: Optional[array[float, 5]] = field(default=None, metadata={"id": 19})
