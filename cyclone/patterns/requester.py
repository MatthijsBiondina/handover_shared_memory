import time
from typing import Type, AnyStr

from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

from cantrips.logging.logger import get_logger
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.defaults import CYCLONE_DEFAULTS
from cyclone.idl.defaults.rpc_idl import RPCIdl
from cyclone.idl.defaults.rpc_status_idl import RPCStatus

logger = get_logger()


class Requester:
    def __init__(
        self,
        domain_participant: CycloneParticipant,
        rpc_name: AnyStr,
        idl_dataclass: Type[RPCIdl],
        deadline: float = 30,
    ):
        self.participant = domain_participant
        self.rpc_name = rpc_name
        self.idl_dataclass = idl_dataclass
        self.deadline = deadline

        # Validate that the responder for the RPC service is active
        self.__validate_responder_status(rpc_name, deadline)

        # Define request and response topics
        self.request_topic = Topic(
            self.participant,
            f"{rpc_name}_request",
            idl_dataclass.Request,
            qos=CYCLONE_DEFAULTS.QOS_RPC,
        )
        self.response_topic = Topic(
            self.participant,
            f"{rpc_name}_response",
            idl_dataclass.Response,
            qos=CYCLONE_DEFAULTS.QOS_RPC,
        )

        # Initialize DDS reader and writer
        self.reader = DataReader(
            self.participant, self.response_topic, qos=CYCLONE_DEFAULTS.QOS_RPC
        )
        self.writer = DataWriter(
            self.participant, self.request_topic, qos=CYCLONE_DEFAULTS.QOS_RPC
        )

    def __validate_responder_status(self, rpc_name: AnyStr, deadline: float):
        # Define topics for status/response
        status_request_topic = Topic(
            self.participant,
            f"{rpc_name}_status_request",
            RPCStatus.Request,
            qos=CYCLONE_DEFAULTS.QOS_RPC_STATUS,
        )
        status_response_topic = Topic(
            self.participant,
            f"{rpc_name}_status_response",
            RPCStatus.Response,
            qos=CYCLONE_DEFAULTS.QOS_RPC_STATUS,
        )

        # Send a status request
        writer = DataWriter(self.participant, status_request_topic)
        time.sleep(1)
        writer.write(RPCStatus.Request(time.time()))

        # Wait for a response within the deadline
        reader = DataReader(
            self.participant, status_response_topic, qos=CYCLONE_DEFAULTS.QOS_RPC
        )
        t_start = time.time()
        while time.time() < t_start + deadline:
            time.sleep(0.1)
            if len(reader.take()):
                return

        # If no response, log an error and raise an exception
        error_message = f'RPC "{rpc_name}" did not respond in time. Is it active?'
        logger.error(error_message)
        raise RuntimeError(error_message)

    def __call__(self, req: RPCIdl.Request | None = None) -> RPCIdl.Response:
        req = (
            req
            if req is not None
            else self.idl_dataclass.Request(timestamp=time.time())
        )
        self.writer.write(req)

        t_start = time.time()
        while time.time() < t_start + self.deadline:
            for response in self.reader.read():
                if response.timestamp == req.timestamp:
                    return response

        error_message = f'RPC "{self.rpc_name}" did not respond in time. Is it active?'
        logger.error(error_message)
        raise RuntimeError(error_message)
