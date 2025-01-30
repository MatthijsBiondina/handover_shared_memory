from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate


@dataclass
@annotate.final
@annotate.autoid("sequential")
class EnvironmentPressure(idl.IdlStruct, typename="Sequence"):
    pressure: float
    temperature: float
