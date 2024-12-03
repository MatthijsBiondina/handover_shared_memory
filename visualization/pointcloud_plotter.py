import numpy as np
from cantrips.configs import load_config
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.sm_reader import SMReader
from visualization.pointcloudstreamer import PointCloudStreamer

logger = get_logger()

class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )

class PointCloudPlotter:
    def __init__(self, participant: CycloneParticipant, streamer: PointCloudStreamer):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.streamer = streamer

    def run(self):
        while True:
            # Read point cloud data
            pointcloud: PointsIDL = self.readers.points()
            not_nan_mask = ~np.any(np.isnan(pointcloud.points), axis=2)
            points_xyz = pointcloud.points[not_nan_mask]
            points_bgr = pointcloud.color[not_nan_mask]

            distance_mask = np.linalg.norm(points_xyz, axis=-1) < 2.5
            points_xyz = points_xyz[distance_mask]
            points_bgr = points_bgr[distance_mask]

            # Downsample the point cloud if necessary
            max_points = 10000  # Adjust as needed
            if points_xyz.shape[0] > max_points:
                indices = np.random.choice(points_xyz.shape[0], max_points, replace=False)
                points_xyz = points_xyz[indices]
                points_bgr = points_bgr[indices]

            points_rgb = (points_bgr.astype(np.float32) / 255.0)

            # Update the streamer with the new point cloud
            self.streamer.update_pointcloud(points_xyz, points_rgb)

            # Sleep or wait for the next point cloud
            self.participant.sleep()

if __name__ == '__main__':
    participant = CycloneParticipant()
    streamer = PointCloudStreamer()
    plotter = PointCloudPlotter(participant, streamer)

    # Start the point cloud reader in a separate thread
    import threading
    threading.Thread(target=plotter.run).start()

    # Run the Flask app (the streamer) in the main thread
    streamer.run()
