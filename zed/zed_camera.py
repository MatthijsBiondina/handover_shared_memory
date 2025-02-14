import logging
from pathlib import Path
import sys
import time

import numpy as np
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
import pyzed.sl as sl

from cyclone.idl_shared_memory.zed_idl import ZEDIDL
from cyclone.patterns.sm_writer import SMWriter
from visualization.webimagestreamer import WebImageStreamer


logger = get_logger()


class Writers:
    def __init__(self, participant):
        self.frame = SMWriter(
            participant,
            CYCLONE_NAMESPACE.ZED_FRAME,
            idl_dataclass=ZEDIDL(),
        )


class ZED:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.zed, self.extrinsics, self.intrinsics = self.init_zed()
        self.writers = Writers(participant)
        
        logger.info("ZED: Ready!")

    def init_zed(self):
        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER

        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error("Cannot connect to ZED")
            sys.exit(1)

        # Load extrinsics
        path = Path(__file__).parent / "extrinsics.npy"
        try:
            extrinsics = np.load(path)
        except FileNotFoundError:
            logger.warning("ZED: No extrinsics calibration found")
            extrinsics = np.eye(4)

        # Get camera calibration parameters
        calibration_params = (
            zed.get_camera_information().camera_configuration.calibration_parameters
        )

        # Access intrinsics
        fx = calibration_params.left_cam.fx  # Focal length x
        fy = calibration_params.left_cam.fy  # Focal length y
        cx = calibration_params.left_cam.cx  # Principal point x
        cy = calibration_params.left_cam.cy  # Principal point y

        # Get intrinsics as matrix
        intrinsics = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        return zed, extrinsics, intrinsics

    def run(self):
        image = sl.Mat()
        depth = sl.Mat()

        while True:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                img = image.get_data()[..., :3]
                dep = depth.get_data()

                msg = ZEDIDL(
                    timestamp=np.array([time.time()]),
                    color=img,
                    depth=dep,
                    extrinsics=self.extrinsics,
                    intrinsics=self.intrinsics,
                )
                self.writers.frame(msg)
            self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ZED(participant)
    node.run()
