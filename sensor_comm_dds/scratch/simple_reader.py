from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
from cyclonedds.idl.types import sequence
from cyclonedds.domain import DomainParticipant, Domain
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader

@dataclass
@annotate.final
@annotate.autoid("sequential")
class Sequence(idl.IdlStruct, typename="Sequence"):
    values: sequence[float]


class Visualiser:
    def __init__(self, topic_data_type, topic_name=None):
        self.topic_name = topic_name
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, self.topic_name, topic_data_type)
        self.reader = DataReader(domain_participant, topic)  # you can also supply Subscriber
        # instead of DomainParticipant to the DataReader constructor, but is not necessary

    def run(self):
        raise NotImplementedError

visualiser = Visualiser(topic_data_type=Sequence, topic_name="test")

print("Trying to read")
for sample in visualiser.reader.take_iter():
    print("Loop")
    print(sample.values)
