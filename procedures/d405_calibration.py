"""
This module implements a calibration procedure for a UR5e robotic arm equipped with a D405 camera.

The calibration involves moving the robot arm in a spiral trajectory, capturing images with the camera, detecting a Charuco board in the images, and computing the extrinsic calibration between the camera and the robot's end effector.

Classes:
    Readers: Manages shared memory readers for data streams.
    CharucoMeasurement: Dataclass for storing a pairing of Charuco board pose and TCP pose.
    CalibrationProcedure: Orchestrates the calibration process by moving the robot, capturing frames, detecting Charuco boards, and computing extrinsics.

Usage:
    Run this module directly to perform the calibration procedure.
"""

import os
import shutil
import time
from dataclasses import dataclass
from random import shuffle
from typing import List

import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import detect_charuco_board
from airo_camera_toolkit.calibration.compute_calibration import (
    eye_in_hand_pose_estimation,
)

from cantrips.debugging.terminal import pyout, pbar, poem
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.patterns.sm_reader import SMReader
from ur5e.ur5e_client import Ur5eClient

logger = get_logger()


class Readers:
    """
    Manages shared memory readers for different data streams.

    Attributes:
        frame (SMReader): Shared memory reader for camera frames.
    """

    def __init__(self, domain_participant: CycloneParticipant):
        """
        Initializes the Readers with shared memory readers.

        Args:
            domain_participant (CycloneParticipant): The domain participant for DDS communication.
        """
        # Initialize the shared memory reader for camera frames
        self.frame = SMReader(
            domain_participant=domain_participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )


@dataclass
class CharucoMeasurement:
    """
    Dataclass for storing a measurement consisting of a Charuco board pose and a TCP (Tool Center Point) pose.

    Attributes:
        charuco_pose (np.ndarray): The pose of the Charuco board in camera coordinates.
        tcp_pose (np.ndarray): The pose of the robot's TCP (end effector) in base coordinates.
    """

    charuco_pose: np.ndarray
    tcp_pose: np.ndarray


class D405CalibrationProcedure:
    """
    Orchestrates the calibration procedure for the robot and camera system.

    Attributes:
        participant (CycloneParticipant): The DDS domain participant.
        readers (Readers): Manages shared memory readers.
        extrinsics (np.ndarray): The current estimate of the extrinsic transformation between camera and end effector.
        ur5e (Ur5eClient): Client for controlling the UR5e robot arm.

    Constants:
        START_JS (List[float]): Starting joint angles for the robot in degrees.
        T_CAM_EE_PATH (str): Path to the file where the extrinsic calibration matrix is stored.
    """

    # Starting joint configuration (in degrees)
    START_JS = [90, -90, 90, -90, -90, -90]
    # Path to the extrinsics matrix file
    T_CAM_EE_PATH = f"{os.path.dirname(__file__)}/../d405/extrinsics.npy"

    def __init__(self, domain_participant: CycloneParticipant):
        """
        Initializes the calibration procedure.

        Args:
            domain_participant (CycloneParticipant): The domain participant for DDS communication.
        """
        self.participant = domain_participant
        # Initialize shared memory readers
        self.readers = Readers(domain_participant)
        # Load existing extrinsics matrix
        try:
            self.extrinsics = np.load(self.T_CAM_EE_PATH)
        except FileNotFoundError:
            self.extrinsics = np.eye(4)
        # Initialize the UR5e robot client
        self.ur5e = Ur5eClient(self.participant)

    def run(self):
        """
        Executes the calibration procedure.

        Steps:
            1. Moves the robot to the starting joint configuration.
            2. Performs a spiral motion to collect frames.
            3. Computes Charuco board poses from the collected frames.
            4. Calibrates the extrinsics matrix using the collected poses.
            5. Saves the new extrinsics matrix to a file.
        """
        # Move robot to the starting joint configuration
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        # Collect frames while moving in a spiral trajectory
        frames = self.calibration_trajectory()

        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        # Detect Charuco board poses from the collected frames
        charuco_poses = self.compute_charuco_poses(frames)
        # Calibrate the extrinsics matrix
        new_extrinsics = self.calibrate_extrinsics_matrix(charuco_poses)
        # Save the new extrinsics matrix
        np.save(self.T_CAM_EE_PATH, new_extrinsics)

    def calibration_trajectory(self):
        frames = []
        focus = np.array([0.1, 0.5, 0.0])
        positions = np.array(
            [
                [-0.1, 0.6, 0.4],
                [-0.1, 0.5, 0.4],
                [-0.1, 0.4, 0.4],
                [-0.1, 0.3, 0.4],
                [-0.0, 0.3, 0.4],
                [+0.1, 0.3, 0.4],
                [+0.2, 0.3, 0.4],
                [+0.2, 0.3, 0.3],
                [+0.1, 0.3, 0.3],
                [+0.1, 0.4, 0.3],
                [+0.0, 0.4, 0.3],
                [-0.1, 0.4, 0.3],
                [-0.1, 0.5, 0.3],
                [-0.1, 0.6, 0.3],
                [-0.1, 0.6, 0.2],
                [-0.1, 0.5, 0.2],
                [-0.1, 0.4, 0.2],
                [-0.0, 0.4, 0.2],
                [+0.1, 0.4, 0.2],
                [+0.2, 0.4, 0.2],
                [+0.15, 0.45, 0.1],
                [+0.1, 0.45, 0.1],
            ]
        )

        for position in pbar(positions, desc="Collecting Frames"):
            tcp = self.ur5e.look_at(position, focus)
            self.ur5e.move_to_tcp_pose(tcp, wait=True)
            time.sleep(0.1)
            frame = self.readers.frame()
            if frame is not None:
                frames.append(frame)

        return frames

    def compute_charuco_poses(self, frames: List[FrameIDL]):
        """
        Detects Charuco boards in the collected frames and computes their poses.

        Args:
            frames (List[FrameIDL]): List of frames collected during the spiral motion.

        Returns:
            List[CharucoMeasurement]: List of measurements containing Charuco board poses and corresponding TCP poses.

        Raises:
            RuntimeError: If not enough Charuco boards are detected.
        """
        charuco_measurements = []
        for frame in frames:
            # Detect the Charuco board in the frame and compute its pose
            charuco_pose = detect_charuco_board(
                image=frame.color, camera_matrix=frame.intrinsics
            )
            # Compute the TCP pose at the time of the frame
            tcp_pose = frame.extrinsics @ np.linalg.inv(self.extrinsics)

            if isinstance(charuco_pose, np.ndarray):
                # Store the measurement if detection was successful
                charuco_measurements.append(CharucoMeasurement(charuco_pose, tcp_pose))

        return charuco_measurements

    def calibrate_extrinsics_matrix(self, measurements: List[CharucoMeasurement]):
        """
        Calibrates the extrinsic transformation matrix between the camera and the end effector.

        Args:
            measurements (List[CharucoMeasurement]): List of Charuco measurements.

        Returns:
            np.ndarray: The calibrated extrinsics matrix.
        """
        extrinsics, error = self.__calibrate_once(measurements)

        logger.info(f"Finished calibration with error: {error:.3f}")

        return extrinsics

    def __calibrate_once(self, measurements: List[CharucoMeasurement]):
        """
        Performs a single calibration iteration using the provided measurements.

        Args:
            measurements (List[CharucoMeasurement]): List of Charuco measurements.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the extrinsics matrix and the associated error.
        """
        tmp_dir = "/tmp/charuco_calibration"
        # Clean up and create temporary directory for calibration files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.mkdir(tmp_dir)

        # Shuffle measurements to randomize selection
        shuffle(measurements)

        # Leave one out in case of noisy measurement
        M = [
            measurements[:ii] + measurements[ii + 1 :]
            for ii in range(len(measurements))
        ]
        extrinsics, error = None, np.inf
        for buffer in M:
            extrinsics_, error_ = eye_in_hand_pose_estimation(
                tcp_poses_in_base=[m.tcp_pose for m in buffer],
                board_poses_in_camera=[m.charuco_pose for m in buffer],
            )
            if error_ is not None:
                if error_ < error:
                    extrinsics = extrinsics_
                    error = error_

        return extrinsics, error


if __name__ == "__main__":
    # Create a DDS participant
    participant = CycloneParticipant()
    # Initialize the calibration procedure
    node = D405CalibrationProcedure(participant)
    # Run the calibration
    node.run()
