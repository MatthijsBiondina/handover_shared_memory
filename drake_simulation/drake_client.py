import time

from cantrips.configs import load_config
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.simulation.world_config_rpc_sample import WorldConfigRPC
from cyclone.patterns.requester import Requester
from drake_simulation.drake_world_config import DrakeWorldConfig


class DrakeClient:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant

    @property
    def world_config(self) -> DrakeWorldConfig:
        requester = Requester(
            domain_participant=self.participant,
            rpc_name=CYCLONE_NAMESPACE.WORLD_CONFIG,
            idl_dataclass=WorldConfigRPC,
        )
        request = WorldConfigRPC.Request(timestamp=time.time())
        response = requester(request)
        return DrakeWorldConfig.from_cyclonedds_response(response)
