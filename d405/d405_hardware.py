import os
import time

import numpy as np
from cyclonedds.domain import DomainParticipant

from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
import pyrealsense2 as rs2

from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_writer import SMWriter

T_CAM_EE_path = f"{os.path.dirname(__file__)}/extrinsics.npy"


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.tcp_pose = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class Writers:
    def __init__(self, participant: DomainParticipant):
        self.frame = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )


logger = get_logger()


class D405:
    def __init__(self, paticipant: CycloneParticipant):
        self.participant = paticipant
        self.readers = Readers(self.participant)
        self.writers = Writers(self.participant)

        self.pipeline, self.align, self.T_cam_ee, self.intrinsics = self.__init_d405()

        logger.info("D405: Ready!")

    def __init_d405(self):
        pipeline = rs2.pipeline()
        config = rs2.config()
        config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 15)
        config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 15)
        profile = pipeline.start()
        device = profile.get_device()
        sensors = device.query_sensors()
        for sensor in sensors:
            if sensor.is_color_sensor():
                sensor.set_option(rs2.option.enable_auto_exposure, 0)
                sensor.set_option(rs2.option.exposure, 5000)
                sensor.set_option(rs2.option.gain, 64)
                sensor.set_option(rs2.option.brightness, 64)

        align = rs2.align(rs2.stream.color)

        intrinsics_object = (
            profile.get_stream(rs2.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        fx = intrinsics_object.fx
        fy = intrinsics_object.fy
        ppx = intrinsics_object.ppx
        ppy = intrinsics_object.ppy
        intrinsics_matrix = np.array(
            [
                [fy, 0.0, ppy],
                [0.0, fx, 848-ppx],
                [0.0, 0.0, 1.0],
            ]
        )
        try:
            extrinsics = np.load(T_CAM_EE_path)
        except FileNotFoundError:
            extrinsics = np.eye(4)
        return pipeline, align, extrinsics, intrinsics_matrix

    def run(self):
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    raise WaitingForFirstMessageException
                depth_image = np.asanyarray(depth_frame.get_data()).T[::-1]
                color_image = np.asanyarray(color_frame.get_data()).transpose(1,0,2)[::-1]

                T_ee_world = np.array(self.readers.tcp_pose().pose)
                T_cam_world = T_ee_world @ self.T_cam_ee

                msg = FrameIDL(
                    timestamp=np.array([time.time()]),
                    color=color_image,
                    depth=depth_image,
                    extrinsics=T_cam_world,
                    intrinsics=self.intrinsics,
                )
                self.writers.frame(msg)
            except WaitingForFirstMessageException:
                pass
            except AttributeError:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = D405(participant)
    node.run()
