from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
from cyclonedds.idl.types import sequence


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Clotheshanger(idl.IdlStruct, typename="Clotheshanger"):
    taxel_values: sequence[int, 4]
