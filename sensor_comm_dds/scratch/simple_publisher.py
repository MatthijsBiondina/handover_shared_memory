import time
import argparse
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.topic import Topic
from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
from cyclonedds.idl.types import sequence

@dataclass
@annotate.final
@annotate.autoid("sequential")
class Sequence(idl.IdlStruct, typename="Sequence"):
    values: sequence[float]


class DataPublisher:
    def __init__(self, topic_name: str, topic_data_type):
        """
        This class publishes data to one or more cycloneDDS topics
        :param topic_data_type: List of data types of the data to be published to the each topic
        """
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, topic_name, topic_data_type)
        publisher = Publisher(domain_participant)
        self.dds_writer = DataWriter(publisher, topic)

    def publish_sensor_data(self, data):
        self.dds_writer.write(data)


publisher = DataPublisher(topic_name="test", topic_data_type=Sequence)

data = Sequence([505])
while True:
    publisher.publish_sensor_data(data)
    print("Published data")
    time.sleep(1)
