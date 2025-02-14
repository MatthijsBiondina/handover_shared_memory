from dataclasses import dataclass
from pathlib import Path
import sys
import time
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import detect_charuco_board
from airo_camera_toolkit.calibration.compute_calibration import (
    eye_to_hand_pose_estimation,
)
from cantrips.debugging.terminal import pbar
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.idl_shared_memory.zed_idl import ZEDIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from ur5e.ur5e_client import Ur5eClient


logger = get_logger()


class Readers:
    def __init__(self, domain_participant: CycloneParticipant):
        self.frame = SMReader(
            domain_participant=domain_participant,
            topic_name=CYCLONE_NAMESPACE.ZED_FRAME,
            idl_dataclass=ZEDIDL(),
        )
        self.tcp = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


@dataclass
class CharucoMeasurement:
    charuco_pose: np.ndarray
    tcp_pose: np.ndarray


class ZEDCalibrationProcedure:
    START_JS = [180, -180, 90, 90, 0, 0]

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.ur5e = Ur5eClient(participant)
        self.initialize()

    def initialize(self):
        self.ur5e.wait_for_planner_initialization()

    def run(self):
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        frames, tcps = self.calibration_trajectory()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)

        charuco_poses = self.compute_charuco_poses(frames, tcps)

        extrinsics, error = eye_to_hand_pose_estimation(
            tcp_poses_in_base=[p.tcp_pose for p in charuco_poses],
            board_poses_in_camera=[p.charuco_pose for p in charuco_poses],
        )
        logger.info(f"Error: {error:.3f}")
        path = Path(__file__).parent.parent / "zed" / "extrinsics.npy"
        np.save(path, extrinsics)

    def calibration_trajectory(self):
        tcps_tgt = np.array(
            [
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, -0.05],
                    [0.0, 1.0, 0.0, 0.45],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.05],
                    [0.0, 1.0, 0.0, 0.45],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.15],
                    [0.0, 1.0, 0.0, 0.4],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.25],
                    [0.0, 1.0, 0.0, 0.4],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.35],
                    [0.0, 1.0, 0.0, 0.35],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.35],
                    [0.0, 1.0, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.25],
                    [0.0, 1.0, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.15],
                    [0.0, 1.0, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -0.35],
                    [0.0, 0.0, -1.0, 0.05],
                    [0.0, 1.0, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ]
        )

        frames = []
        tcps_measured = []
        for pose in pbar(tcps_tgt, desc="Collecting Frames"):
            self.ur5e.move_to_tcp_pose(pose, wait=True)
            time.sleep(1)

            while True:
                frame = self.readers.frame()
                tcp = self.readers.tcp()
                if frame is not None and tcp is not None:
                    frames.append(frame)
                    tcps_measured.append(tcp)
                    break
                self.participant.sleep()

        return frames, tcps_measured

    def compute_charuco_poses(self, frames, tcps):
        charuco_measurements = []
        for frame, tcp in zip(frames, tcps):
            charuco_pose = detect_charuco_board(
                frame.color, camera_matrix=frame.intrinsics
            )
            charuco_measurements.append(CharucoMeasurement(charuco_pose, tcp.pose))
        return charuco_measurements


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ZEDCalibrationProcedure(participant)
    node.run()
