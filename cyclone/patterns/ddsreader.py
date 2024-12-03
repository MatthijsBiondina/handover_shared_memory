from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlMeta
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

from cantrips.debugging.terminal import pyout
from cyclone.defaults import CYCLONE_DEFAULTS


class DDSReader:
    def __init__(
        self,
        domain_participant: DomainParticipant,
        topic_name: str,
        idl_dataclass: IdlMeta,
    ):
        self.topic = Topic(domain_participant, topic_name, idl_dataclass)
        self.reader = DataReader(domain_participant, self.topic, CYCLONE_DEFAULTS.QOS)

    def __call__(self):
        data = self.reader.read()
        if len(data) == 0:
            return None
        return data[-1]

    def take(self):
        data = self.reader.take()
        if len(data) == 0:
            return None
        return data[-1]
