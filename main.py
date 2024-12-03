from cyclonedds.domain import DomainParticipant

from cyclone.pubsub_pattern.publisher import Publisher


def main():
    participant = DomainParticipant()
    publisher = Publisher(participant)