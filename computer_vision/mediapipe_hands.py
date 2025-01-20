from dataclasses import dataclass
from time import time
from typing import Dict
import mediapipe
import numpy as np
import torch

from cantrips.configs import load_config
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.defaults import Config
from cyclone.idl.mediapipe.hand_sample import HandSample
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.hand = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_HAND,
            idl_dataclass=HandSample,
        )


logger = get_logger()


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    u: int
    v: int


class MediapipeHands:
    TRACKED_LANDMARKS = [
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    def __init__(self, domain_participant: CycloneParticipant):
        self.config = load_config()
        self.participant = domain_participant
        self.readers = Readers(domain_participant)
        self.writers = Writers(domain_participant)

        self.mp_draw = mediapipe.solutions.drawing_utils
        self.mp_hands = mediapipe.solutions.hands
        self.model = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )

        self.web_streamer = WebImageStreamer(title="hand", port=5007)

        logger.info(f"MediapipeHands: Ready!")

    def run(self):
        while True:
            img = np.zeros((Config.height, Config.width, 3), dtype=np.uint8)
            try:
                points = self.readers.points()
                if points is not None:
                    img = points.color
                target = self.readers.target()
                if points is None or target is None:
                    raise ContinueException

                hands = self.get_hand_keypoints(points)
                hand = self.filter_by_distance(hands, target)
                self.publish(hand)

                img = self.draw_keypoints(hand, img)

            except ContinueException:
                pass
            finally:
                self.web_streamer.update_frame(img[..., ::-1])
                self.participant.sleep()

    def get_hand_keypoints(self, points: PointsIDL):
        img = points.color
        h, w, _ = img.shape
        results = self.model.process(img)
        if not results.multi_hand_landmarks:
            raise ContinueException

        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand = {}
            for landmark in self.TRACKED_LANDMARKS:
                index = eval(f"self.mp_hands.HandLandmark.{landmark}")
                u = np.clip(int(round(hand_landmarks.landmark[index].x * w)), 0, w - 1)
                v = np.clip(int(round(hand_landmarks.landmark[index].y * h)), 0, h - 1)
                hand[landmark] = Landmark(
                    x=points.points[v, u, 0],
                    y=points.points[v, u, 1],
                    z=points.points[v, u, 2],
                    u=u,
                    v=v,
                )
            hands.append(hand)

        return hands

    def filter_by_distance(
        self,
        hands: Dict[str, Landmark],
        target: CoordinateSample,
        distance_threshold=0.1,
    ):
        D = lambda l: np.linalg.norm(
            np.array([l.x, l.y, l.z]) - np.array([target.x, target.y, target.z])
        )
        # Focus on hand closest to target
        hand = min(
            hands,
            key=lambda h: np.nanmedian([D(l) for l in h.values()]),
        )

        for key, landmark in hand.items():
            if D(landmark) < distance_threshold:
                continue
            hand[key] = None

        return hand

    def publish(self, hand):
        def map_to_array(landmark: Landmark):
            if landmark is None:
                return None
            return [
                landmark.x,
                landmark.y,
                landmark.z,
                float(landmark.u),
                float(landmark.z),
            ]

        msg = HandSample(
            timestamp=time(),
            thumb_mcp=map_to_array(hand.get("THUMB_MCP")),
            thumb_ip=map_to_array(hand.get("THUMB_IP")),
            thumb_tip=map_to_array(hand.get("THUMB_TIP")),
            index_finger_mcp=map_to_array(hand.get("INDEX_FINGER_MCP")),
            index_finger_pip=map_to_array(hand.get("INDEX_FINGER_PIP")),
            index_finger_dip=map_to_array(hand.get("INDEX_FINGER_DIP")),
            index_finger_tip=map_to_array(hand.get("INDEX_FINGER_TIP")),
            middle_finger_mcp=map_to_array(hand.get("MIDDLE_FINGER_MCP")),
            middle_finger_pip=map_to_array(hand.get("MIDDLE_FINGER_PIP")),
            middle_finger_dip=map_to_array(hand.get("MIDDLE_FINGER_DIP")),
            middle_finger_tip=map_to_array(hand.get("MIDDLE_FINGER_TIP")),
            ring_finger_mcp=map_to_array(hand.get("RING_FINGER_MCP")),
            ring_finger_pip=map_to_array(hand.get("RING_FINGER_PIP")),
            ring_finger_dip=map_to_array(hand.get("RING_FINGER_DIP")),
            ring_finger_tip=map_to_array(hand.get("RING_FINGER_TIP")),
            pinky_mcp=map_to_array(hand.get("PINKY_MCP")),
            pinky_pip=map_to_array(hand.get("PINKY_PIP")),
            pinky_dip=map_to_array(hand.get("PINKY_DIP")),
            pinky_tip=map_to_array(hand.get("PINKY_TIP")),
        )
        self.writers.hand(msg=msg)


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = MediapipeHands(participant)
    with torch.no_grad():
        node.run()
