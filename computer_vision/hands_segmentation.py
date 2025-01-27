import os
import sys
import warnings

import numpy as np
from cantrips.configs import load_config
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.masks_idl import MasksIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
import torch
import torch.hub
import torch.nn as nn
import torchvision.transforms as T

from cyclone.patterns.sm_writer import SMWriter
from visualization.webimagestreamer import WebImageStreamer


"""
@article{camporese2021HandsSeg,
  title   = "Hands Segmentation is All You Need",
  author  = "Camporese, Guglielmo",
  journal = "https://github.com/guglielmocamporese",
  year    = "2021",
  url     = "https://github.com/guglielmocamporese/hands-segmentation-pytorch"
}
"""


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
        self.masks = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.SAM_MASKS,
            idl_dataclass=MasksIDL(),
        )


logger = get_logger()


class HandSegmentation:
    MAX_DISTANCE = 0.1

    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        self.device = torch.device("cuda:0")
        self.dnn_model, self.transform = self.load_model()

        logger.info("HandSegmentation Ready!")

    def load_model(self):
        # First ensure repository is downloaded
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*'pretrained' is deprecated.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
            torch.hub.load(
                "guglielmocamporese/hands-segmentation-pytorch",
                "hand_segmentor",
                pretrained=False,  # Don't load pretrained weights yet
                force_reload=False,
            )

            # Get the hub directory where the model files are downloaded
            hub_dir = torch.hub.get_dir()
            repo_dir = os.path.join(
                hub_dir, "guglielmocamporese_hands-segmentation-pytorch_master"
            )
            model_path = os.path.join(repo_dir, "model.py")

            # Import the module and get the model class
            import importlib.util

            spec = importlib.util.spec_from_file_location("model_module", model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            # Create and load the model from checkpoint
            model = model_module.HandSegModel.load_from_checkpoint(
                "checkpoint/checkpoint.ckpt", map_location=self.device
            )
            model.to(self.device)
            model.eval()

            transform = T.Compose(
                [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )

        return model, transform

    def run(self):
        while True:
            try:
                points: PointsIDL = self.readers.points()
                target: CoordinateSample = self.readers.target()
                if points is None or target is None:
                    raise ContinueException

                dnn_hand_mask = self.apply_dnn_model(points.color)
                distance_mask = self.distance_mask(points, target)

                mask_hand = distance_mask & dnn_hand_mask
                mask_object = distance_mask & ~dnn_hand_mask

                msg = MasksIDL(
                    timestamp=points.timestamp,
                    color=points.color,
                    depth=points.depth,
                    points=points.points,
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

    @torch.no_grad()
    def apply_dnn_model(self, img: np.ndarray):
        X = torch.tensor(img, dtype=torch.float32).to(self.device)
        X /= 255.0
        X = X.permute(2, 0, 1)
        X = self.transform(X)
        X = X[None, ...]
        y = self.dnn_model(X)
        m = y.argmax(1).squeeze().cpu().numpy().astype(np.bool_)
        return m

    def distance_mask(self, points: PointsIDL, target: CoordinateSample):
        tgt = torch.tensor(
            np.array([target.x, target.y, target.z]), dtype=torch.float32
        ).to(self.device)
        pts = torch.tensor(points.points, dtype=torch.float32).to(self.device)
        D = torch.linalg.norm(pts - tgt[None, None, :], dim=-1)
        mask = D < self.MAX_DISTANCE
        return mask.cpu().numpy()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = HandSegmentation(participant)
    node.run()
