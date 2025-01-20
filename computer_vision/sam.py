import time
import numpy as np
import torch
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.defaults import Config
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl_shared_memory.masks_idl import MasksIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader

from segment_anything import SamPredictor, sam_model_registry

from cyclone.patterns.sm_writer import SMWriter

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.masks = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.SAM_MASKS,
            idl_dataclass=MasksIDL(),
        )


class SegmentAnything:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant=participant)
        self.writers = Writers(participant=participant)
        self.sam = sam_model_registry["vit_b"](
            checkpoint="/home/matt/System/SAM/vit_b.pth"
        )
        self.predictor = SamPredictor(self.sam)
        self.device = torch.device("cuda")

        # placeholders
        self.xrange = torch.arange(Config.width).to(self.device)
        self.yrange = torch.arange(Config.height).to(self.device)

        logger.info("SegmentAnything: Ready!")

    def run(self):
        while True:
            try:
                target = self.readers.target()
                points = self.readers.points()
                if target is None or points is None:
                    raise ContinueException

                t0 = time.time()
                self.predictor.set_image(points.color)
                logger.info(f"SAM set image: {time.time() - t0:.2f}")
                bbox = self.compute_bounding_box(points, target)

                t0 = time.time()
                masks, _, _ = self.predictor.predict(box=np.array(bbox))
                logger.info(f"SAM predict: {time.time() - t0:.2f}s")

                mask_object = masks[0]
                t0 = time.time()
                mask_hand = self.compute_hand_mask(points, target, mask_object)
                logger.info(f"SAM compute: {time.time() - t0:.2f}s")

                msg = MasksIDL(
                    timestamp=points.timestamp,
                    color=points.color,
                    depth=points.depth,
                    mask_hand=mask_hand,
                    mask_object=mask_object,
                    intrinsics=points.intrinsics,
                    extrinsics=points.extrinsics,
                )
                self.writers.masks(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

        logger.info("running")

    @torch.no_grad()
    def compute_bounding_box(self, points, target):
        mask = self.get_distance_mask(points, target)

        x = self.xrange[torch.any(mask, dim=0)]
        y = self.yrange[torch.any(mask, dim=1)]

        if x.shape == 0 or y.shape == 0:
            raise ContinueException

        left = torch.min(x).item()
        right = torch.max(x).item()
        top = torch.min(y).item()
        bottom = torch.max(y).item()

        return left, top, right, bottom

    def compute_hand_mask(self, points, target, object_mask):
        d_mask = self.get_distance_mask(points, target)
        hand_mask = d_mask.cpu().numpy() & ~object_mask
        return hand_mask

    def get_distance_mask(self, points, target):
        tgt = torch.tensor(
            np.array([target.x, target.y, target.z]), dtype=torch.float32
        ).to(self.device)
        pts = torch.tensor(points.points, dtype=torch.float32).to(self.device)
        D = torch.linalg.norm(pts - tgt[None, None, :], dim=-1)
        mask = D < 0.1
        return mask


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = SegmentAnything(participant=participant)
    node.run()
