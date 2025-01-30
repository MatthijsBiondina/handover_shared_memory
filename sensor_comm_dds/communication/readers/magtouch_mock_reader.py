import time
import random
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.topic import Topic


domain_participant = DomainParticipant()
topic = Topic(domain_participant, 'MagTouchMock', MagTouch4)
publisher = Publisher(domain_participant)
writer = DataWriter(publisher, topic)


taxels = [MagTouchTaxel(0, 0, 0) for _ in range(4)]
magtouch = MagTouch4(taxels)


while True:
    magtouch.taxels[random.randint(0, 3)].x = random.random() * 1.5
    magtouch.taxels[random.randint(0, 3)].y = random.random() * 1.5
    magtouch.taxels[random.randint(0, 3)].z = random.random() * 1.5
    writer.write(magtouch)
    print(">> Wrote magtouch")
    time.sleep(random.random() * 0.9 + 0.1)