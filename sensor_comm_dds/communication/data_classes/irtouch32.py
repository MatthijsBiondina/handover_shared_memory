from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
from cyclonedds.idl.types import sequence


@dataclass
@annotate.final
@annotate.autoid("sequential")
class IRTouch32(idl.IdlStruct, typename="IRTouch32"):
    taxel_values: sequence[int, 32]
    strain_value: int
