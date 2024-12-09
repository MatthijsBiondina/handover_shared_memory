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


class CalibrationProcedure:
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
    START_JS = [90, -90, 90, 90, 90, 0]
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
        self.extrinsics = np.load(self.T_CAM_EE_PATH)
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
        frames = self.spiral()
        
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        # Detect Charuco board poses from the collected frames
        charuco_poses = self.compute_charuco_poses(frames)
        # Calibrate the extrinsics matrix
        new_extrinsics = self.calibrate_extrinsics_matrix(charuco_poses)
        # Save the new extrinsics matrix
        np.save(self.T_CAM_EE_PATH, new_extrinsics)

    def spiral(self, X=-0.1, Y=0.3, Z_min=0.1, Z_max=0.5, r=0.15, T=10):
        """
        Moves the robot in a spiral trajectory and collects frames.

        Args:
            X (float): X-coordinate center of the spiral.
            Y (float): Y-coordinate center of the spiral.
            Z_min (float): Minimum Z-coordinate (start height).
            Z_max (float): Maximum Z-coordinate (end height).
            r (float): Radius of the spiral.
            T (float): Total duration of the spiral motion in seconds.

        Returns:
            List[FrameIDL]: List of collected frames during the motion.
        """

        def xyz(t):
            """
            Computes the x, y, z coordinates at time t during the spiral motion.

            Args:
                t (float): Time since the start of the spiral motion.

            Returns:
                Tuple[float, float, float]: The x, y, z coordinates.
            """
            # Normalize progress between 0 and 1
            progress = np.clip(t, 0, T) / T
            # Compute angular position (two full rotations)
            theta = 4 * np.pi * progress
            # Compute x, y positions based on spiral equation
            x = X + r * np.cos(theta)
            y = Y + r * np.sin(theta)
            # Compute z position descending from Z_max to Z_min
            z = Z_max - (Z_max - Z_min) * progress
            return x, y, z

        # Move to the starting position of the spiral
        x, y, z = xyz(0)
        self.ur5e.move_to_tcp_pose(
            np.array(
                [
                    [-1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, -1.0, z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            wait=True,
        )

        t0 = time.time()
        frames = []
        while time.time() - t0 < T:
            t = time.time() - t0
            x, y, z = xyz(t)
            # Command robot to move to the next position (non-blocking)
            self.ur5e.move_to_tcp_pose(
                np.array(
                    [
                        [-1.0, 0.0, 0.0, x],
                        [0.0, 1.0, 0.0, y],
                        [0.0, 0.0, -1.0, z],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
            # Sleep to allow other DDS participants to process
            self.participant.sleep()
            # Get the latest frame from the camera
            frame: FrameIDL = self.readers.frame()
            # Ensure the frame is new and append it to the list
            if frame is not None and (
                len(frames) == 0 or frame.timestamp[0] > frames[-1].timestamp[0]
            ):
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

        if len(charuco_measurements) < 100:
            # If not enough measurements were collected, raise an error
            raise RuntimeError("Cannot see Charuco board well enough.")

        return charuco_measurements

    def calibrate_extrinsics_matrix(self, measurements: List[CharucoMeasurement]):
        """
        Calibrates the extrinsic transformation matrix between the camera and the end effector.

        Args:
            measurements (List[CharucoMeasurement]): List of Charuco measurements.

        Returns:
            np.ndarray: The calibrated extrinsics matrix.
        """
        extrinsics, error = None, np.inf
        # Progress bar for calibration iterations
        bar = pbar(range(100), desc="Calibrating")
        for _ in bar:
            if error < 0.01:
                break
            # Perform one calibration iteration
            extrinsics_, error_ = self.__calibrate_once(measurements)
            if error_ < error:
                # Update the best extrinsics matrix and error
                extrinsics, error = extrinsics_, error_
                # Update progress bar description with current error
                bar.desc = poem(f"Calibrating {error:.3f}")

        logger.warning(f"Finished calibration with error: {error:.3f}")

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
        # Initialize buffer with first two measurements
        buffer = measurements[:2]
        extrinsics, error = None, np.inf
        # Progress bar over the remaining measurements
        bar = pbar(range(2, len(measurements)))
        for ii in bar:
            # Perform eye-in-hand pose estimation using current buffer and new measurement
            extrinsics_, error_ = eye_in_hand_pose_estimation(
                tcp_poses_in_base=[m.tcp_pose for m in buffer]
                + [measurements[ii].tcp_pose],
                board_poses_in_camera=[m.charuco_pose for m in buffer]
                + [measurements[ii].charuco_pose],
            )
            if error_ is None:
                # Skip if estimation failed
                continue

            if error_ < error:
                # Update best extrinsics and error
                error = error_
                extrinsics = extrinsics_
                # Add the current measurement to the buffer
                buffer.append(measurements[ii])
                # Update progress bar description with current error
                bar.desc = poem(f"           ({error:.3f})")
        if len(buffer) >= 10:
            return extrinsics, error
        else:
            return extrinsics, np.inf


if __name__ == "__main__":
    # Create a DDS participant
    participant = CycloneParticipant()
    # Initialize the calibration procedure
    node = CalibrationProcedure(participant)
    # Run the calibration
    node.run()
