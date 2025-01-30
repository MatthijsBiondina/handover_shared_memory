from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
from cyclonedds.idl.types import sequence
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MagTouch4(idl.IdlStruct, typename="MagTouch2x2"):
    # TODO: consider not using taxels, this adds unpacking overhead, just do a matrix
    taxels: sequence[MagTouchTaxel, 4]
