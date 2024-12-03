import cv2
import numpy as np
from cyclonedds.domain import DomainParticipant

from cantrips.configs import load_config
from cantrips.exceptions import WaitingForFirstMessageException
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )


class D405Plotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        # Create an instance of WebImageStreamer
        self.web_streamer = WebImageStreamer(title="RealSense D405", port=5000)

    def run(self):
        while True:
            try:
                frame: FrameIDL = self.readers.frame()

                display_image = self.make_depth_overlay(frame)

                # Update the frame in WebImageStreamer
                self.web_streamer.update_frame(display_image)

            except WaitingForFirstMessageException:
                pass
            finally:
                self.participant.sleep()

    def make_depth_overlay(self, frame: FrameIDL):
        color = frame.color
        depth = frame.depth

        # preprocess depth_image
        depth = depth * self.config.depth_scale
        depth[depth > self.config.depth_event_horizon] = 0
        depth[depth == 0] = self.config.depth_event_horizon
        depth = 1 - (depth / self.config.depth_event_horizon)

        # make heatmap
        heatmap = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        # combine heatmap and image
        grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) / 255.0

        combined = (grey[..., None] * heatmap).astype(np.uint8)

        return combined


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = D405Plotter(participant)
    node.run()
