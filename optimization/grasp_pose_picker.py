import time
import numpy as np
import torch
from torch import Tensor, tensor
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.defaults.boolean_idl import StateSample
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.idl_shared_memory.masks_idl import MasksIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader
from optimization.cmaes import CMAES
from procedures.state_machine_states import States

torch.set_printoptions(precision=2, sci_mode=False)


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.masks = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.SAM_MASKS,
            idl_dataclass=MasksIDL(),
        )
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )
        self.state = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.STATE_MACHINE_STATE,
            idl_dataclass=StateSample,
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.pose = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.GRASP_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


logger = get_logger()


class GraspPosePicker:
    POPULATION_SIZE = 100
    N = 100
    DEVICE = "cpu"

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        # self.optimizer = CMAES(dimension=4, population_size=100)
        self.active = False
        self.optimizer = None
        self.device = torch.device(self.DEVICE)

        logger.info("CMAES Ready!")

    def run(self):
        while True:
            try:
                state: StateSample = self.readers.state()
                if state is None:
                    raise ContinueException

                if not state.state == States.REACHING:
                    if self.active:
                        self.active = False
                        self.optimizer = None
                    raise ContinueException

                masks: MasksIDL = self.readers.masks()
                if masks is None:
                    raise ContinueException

                if not self.active:
                    self.init_optimizer(masks)
                    self.active = True

                solution = self.optimizer_step(masks)

                tcp = self.xyzrpy_to_matrix(solution).cpu().numpy()

                msg = TCPPoseSample(
                    timestamp=masks.timestamp,
                    pose=tcp.tolist(),
                    velocity=np.zeros_like(tcp).tolist(),
                )
                self.writers.pose(msg)

            except ContinueException:
                self.participant.sleep()
                pass
            finally:
                # self.participant.sleep()
                pass

    def init_optimizer(self, masks: MasksIDL, dimension=5):
        tgt = masks.points[masks.mask_object]
        if tgt.shape[0] > self.N:
            tgt = tgt[np.random.permutation(tgt.shape[0])[: self.N]]
        tgt = tensor(tgt, device=self.device)
        if tgt.shape[0] == 0:
            raise ContinueException

        mu = torch.zeros(dimension, device=self.device)
        mu[:3] = torch.median(tgt, dim=0).values
        mu[3] = torch.atan2(mu[1], mu[0]) + 0.5 * np.pi

        self.optimizer = CMAES(
            dimension=dimension,
            population_size=256,
            initial_mean=mu,
            initial_sigma=1.0,
            device=self.DEVICE,
        )

    def optimizer_step(self, masks: MasksIDL):
        # Generate new solutions
        solutions = self.optimizer.ask()

        # Evaluate solutions
        fitness_values = self.evaluate(solutions, masks)

        # Update internal model
        self.optimizer.tell(solutions, fitness_values)

        return self.optimizer.mean

    def evaluate(self, solutions: torch.Tensor, masks: MasksIDL):
        tgt_np = masks.points[masks.mask_object]
        obs_np = masks.points[masks.mask_hand]

        if tgt_np.shape[0] > self.N:
            tgt_np = tgt_np[np.random.permutation(tgt_np.shape[0])[: self.N]]
        if obs_np.shape[0] > self.N:
            obs_np = obs_np[np.random.permutation(obs_np.shape[0])[: self.N]]

        tgt = tensor(tgt_np, device=self.device)
        obs = tensor(obs_np, device=self.device)
        if tgt.shape[0] == 0:
            raise ContinueException

        tcp = self.xyzrpy_to_matrix(solutions)
        L_dist = self.distance_loss(tcp, tgt)
        L_angl = self.orientation_loss(tcp)
        L_tgtx = -10 * self.count_nr_of_points_between_fingers(tcp, tgt)
        L_obsx = 30 * self.count_nr_of_points_between_fingers(
            tcp, obs, finger_height=0.06
        )
        L_eigv = 0.1*self.eigen_loss(tcp, tgt)

        # logger.info(
        #     f"D: {L_dist[0]:.2f} | "
        #     f"O: {L_angl[0]:.2f} | "
        #     f"+: {L_tgtx[0]:.2f} | "
        #     f"-: {L_obsx[0]:.2f}"
        # )

        return L_dist + L_tgtx + L_obsx + L_angl + L_eigv

    @staticmethod
    def xyzrpy_to_matrix(xyzrpy: Tensor) -> Tensor:
        """
        Convert xyz position and rpy euler angles to 4x4 homogeneous matrix.
        xyzrpy: tensor [..., 6] containing xyz translation and rpy rotation
        returns: tensor [..., 4, 4] homogeneous transformation matrix
        """
        if xyzrpy.shape[-1] == 4:
            x, y, z, yaw = torch.split(xyzrpy, 1, dim=-1)
            pitch = torch.zeros_like(yaw)
        else:
            x, y, z, yaw, pitch = torch.split(xyzrpy, 1, dim=-1)
        roll = torch.ones_like(yaw) * 0.5 * np.pi

        # Pre-compute trigonometric functions
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        # Build rotation matrix
        R = torch.stack(
            [
                cp * cy,
                -cr * sy + sr * sp * cy,
                sr * sy + cr * sp * cy,
                cp * sy,
                cr * cy + sr * sp * sy,
                -sr * cy + cr * sp * sy,
                -sp,
                sr * cp,
                cr * cp,
            ],
            dim=-1,
        ).reshape(*xyzrpy.shape[:-1], 3, 3)

        # Build translation vector
        t = torch.cat([x, y, z], dim=-1)

        # Construct homogeneous matrix
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=xyzrpy.device)
        bottom_row = bottom_row.expand(*xyzrpy.shape[:-1], 4)

        H = torch.cat(
            [torch.cat([R, t.unsqueeze(-1)], dim=-1), bottom_row.unsqueeze(-2)], dim=-2
        )

        return H

    @staticmethod
    def distance_loss(tcp: Tensor, tgt: Tensor) -> Tensor:
        X = tcp[:, :3, 3]
        tgt_pt = torch.median(tgt, dim=0).values
        distance = torch.linalg.norm(X - tgt_pt[None, :], dim=-1)

        return distance**2

    @staticmethod
    def orientation_loss(tcp: Tensor) -> Tensor:
        tool_forward_vector = tcp[:, :2, 2]
        manipulator_line = tcp[:, :2, 3]

        tool_forward_vector = tool_forward_vector / torch.linalg.norm(
            tool_forward_vector, dim=-1, keepdim=True
        )
        manipulator_line = manipulator_line / torch.linalg.norm(
            manipulator_line, dim=-1, keepdim=True
        )

        loss = torch.linalg.norm(tool_forward_vector - manipulator_line, dim=-1)

        return loss

    @staticmethod
    def count_nr_of_points_between_fingers(
        tcp: Tensor,
        cloud: Tensor,
        maxN=100,
        finger_width=0.03,
        finger_height=0.02,
        finger_distance=0.08,
    ) -> Tensor:
        """
        Count points that fall within the box-shaped area between gripper fingers.

        Args:
            tcp: (bs, 4, 4) End effector poses to be evaluated
            cloud: (N, 3) Points observed by camera
            maxN: Maximum number of points to process
            finger_width: Width of each finger in meters
            finger_height: Height of each finger in meters
            finger_depth: Depth of fingers in meters
            finger_distance: Distance between fingers in meters

        Returns:
            Tensor: (bs,) Number of points between fingers for each pose
        """
        if cloud.shape[0] > maxN:
            cloud = cloud[torch.randperm(cloud.shape[0])[:maxN]]
        if cloud.shape[0] == 0:
            return torch.zeros(tcp.shape[0], device=tcp.device)

        # Transform points to TCP frame for each pose in batch
        bs = tcp.shape[0]
        points = cloud[None, ...]
        points_homog = torch.cat(
            (points, torch.ones((*points.shape[:-1], 1), device=points.device)), dim=-1
        )[
            ..., None
        ]  # (1, 1000, 4, 1)

        # Transform
        tcp_inv = torch.linalg.inv(tcp)[:, None, :, :]
        P = (tcp_inv @ points_homog).squeeze(-1)[..., :3]

        x_mask = (P[..., 0] > -finger_distance / 2) & (P[..., 0] < finger_distance / 2)
        y_mask = (P[..., 1] > -finger_height / 2) & (P[..., 1] < finger_height / 2)
        z_mask = (P[..., 2] > -finger_width) & (P[..., 2] < 0)

        count = torch.sum(x_mask & y_mask & z_mask, dim=-1)
        return count / cloud.shape[0]

    @staticmethod
    def eigen_loss(tcp: Tensor, cloud: Tensor, maxN=100):
        N = cloud.shape[0]
        if N > maxN:
            cloud = cloud[torch.randperm(N)[:maxN]]
            N = maxN
        elif N == 0:
            return torch.zeros(tcp.shape[0], device=tcp.device)

        # Center the points by subtracting the mean
        mean = torch.mean(cloud, dim=0)
        centered = cloud - mean

        # Compute covariance matrix
        cov = torch.matmul(centered.T, centered) / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # First Principal Component of pointcloud
        PC1 = eigenvectors[:, 0] / torch.linalg.norm(eigenvectors[:, 0])
        if PC1[2] < 0:
            PC1 = PC1 * -1  # point up

        # compute cosine similarity between tcp-y and pc1
        tcp_y = tcp[:, :3, 1]

        sim = (tcp_y @ PC1[:, None]).squeeze(-1)

        return 1 - sim


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = GraspPosePicker(participant)
    node.run()
