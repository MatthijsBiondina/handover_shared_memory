from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MagTouchTaxel(idl.IdlStruct, typename="Taxel"):
    x: types.float32
    y: types.float32
    z: types.float32
