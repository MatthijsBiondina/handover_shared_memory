from typing import Dict, List, Any

import airo_models
import numpy as np
import airo_drake
from airo_drake import SingleArmScene
from pydrake.geometry import Box, SceneGraph, Meshcat, Sphere
from pydrake.geometry import Sphere as DrakeSphere
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (
    SpatialInertia,
    ModelInstanceIndex,
    UnitInertia,
    RigidBody,
)
from pydrake.planning import RobotDiagramBuilder, RobotDiagram
from pydrake.systems.framework import Context
from pydrake.common.eigen_geometry import Quaternion
from torch import Tensor

from cantrips.configs import load_config
from cantrips.logging.logger import get_logger
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from drake_simulation.drake_world_config import DrakeWorldConfig
from utils.simulation_utils import matrix2cupose

logger = get_logger()


class DrakeScene:
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, spheres: Tensor | None = None):
        self.config = load_config()

        (
            self.drake_indices,
            self.sphere_indices,
            self.robot_diagram_builder,
            self.meshcat,
            self.plant,
            self.scene_graph,
            self.diagram,
            self.context,
            self.scene,
            self.__ee_triad,
            self.__goal_triad,
        ) = self.__initialize_scene(spheres)

        logger.info("Drake Scene Ready!")

    def __initialize_scene(
        self,
        spheres: Tensor | None = None,
    ) -> (
        Dict[str, Dict[str, ModelInstanceIndex]],
        List[Any],
        RobotDiagramBuilder,
        Meshcat,
        MultibodyPlant,
        SceneGraph,
        RobotDiagram,
        Context,
        SingleArmScene,
        RigidBody,
        RigidBody,
    ):
        # Create a buffer to store model instance indices for world and robot elements.
        index_buffer = {"world": {}, "robot": {}}
        sphere_buffer = []

        # Initialize the robot diagram builder and add Meshcat for visualization.
        robot_diagram_builder = RobotDiagramBuilder()
        meshcat = airo_drake.add_meshcat(robot_diagram_builder)
        meshcat.SetCameraPose(np.array([0.0, 2.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        # Add a manipulator to the scene, which includes the robot arm and gripper.
        # The robot arm is UR5e and the gripper is Robotiq 2F-85.
        # gripper_rotation = RollPitchYaw(0, 0, -np.pi / 2).ToRotationMatrix().matrix()
        # gripper_transform = np.eye(4)
        # gripper_transform[:3, :3] = gripper_rotation
        # pyout(gripper_transform)
        angle = 0
        gripper_transform = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0, 0.0],
                [np.sin(angle), np.cos(angle), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        (
            index_buffer["robot"]["arm"],
            index_buffer["robot"]["gripper"],
        ) = airo_drake.add_manipulator(
            robot_diagram_builder,
            "ur5e",
            "robotiq_2f_85",
            arm_transform=np.eye(4),
            gripper_transform=gripper_transform,
        )

        # Add a floor to the scene and store its index.
        index_buffer["world"]["floor"] = airo_drake.add_floor(
            robot_diagram_builder, x_size=1.8, y_size=2.0
        )
        index_buffer["world"]["wall_R"] = self.__add_wall(
            robot_diagram_builder, -0.8, 0.0, 1.0, 0.2, 2.0, 2.0, "wall_R"
        )
        index_buffer["world"]["wall_L"] = self.__add_wall(
            robot_diagram_builder, 0.8, 0.0, 1.0, 0.2, 2.0, 2.0, "wall_L"
        )
        index_buffer["world"]["wall_B"] = self.__add_wall(
            robot_diagram_builder, 0.0, -0.8, 1.0, 2.0, 0.2, 2.0, "wall_B"
        )

        # Retrieve references to the plant, parser, and scene graph for further configuration.
        plant = robot_diagram_builder.plant()
        parser = robot_diagram_builder.parser()
        scene_graph = robot_diagram_builder.scene_graph()

        ee_triad = self.__add_triad("ee_triad", plant)
        goal_triad = self.__add_triad("goal_triad", plant)

        # Initialize curobo spheres
        if spheres is not None:
            sphere_model_instance = plant.AddModelInstance("spheres")

            for i, sphere in enumerate(spheres):
                radius = sphere[3].cpu().numpy()

                # Define a sphere geometry
                sphere_shape = Sphere(radius)
                sphere_body = plant.AddRigidBody(
                    f"curobo_sphere_{i}",
                    sphere_model_instance,
                    SpatialInertia(
                        mass=0.0,
                        p_PScm_E=np.zeros(3),
                        G_SP_E=UnitInertia(0.0, 0.0, 0.0),
                    ),
                )
                # Attach visual geometry to dummy body
                plant.RegisterVisualGeometry(
                    sphere_body,
                    RigidTransform(),
                    sphere_shape,
                    f"curobo_visual_{i}",
                    np.array([1.0, 0, 0, 0.5]),
                )

                sphere_buffer.append(sphere_body)

        diagram, context = airo_drake.finish_build(robot_diagram_builder, meshcat)
        scene = SingleArmScene(
            diagram,
            index_buffer["robot"]["arm"],
            index_buffer["robot"]["gripper"],
            meshcat,
        )

        # Return the index buffer, robot diagram builder, meshcat visualizer, and the plant.
        return (
            index_buffer,
            sphere_buffer,
            robot_diagram_builder,
            meshcat,
            plant,
            scene_graph,
            diagram,
            context,
            scene,
            ee_triad,
            goal_triad,
        )

    def __add_triad(self, name: str, plant: MultibodyPlant) -> RigidBody:
        triad_model_instance = plant.AddModelInstance(name)

        # Create a dummy rigid body to represent the triad
        triad_body = plant.AddRigidBody(
            name,
            triad_model_instance,
            SpatialInertia(
                mass=0.0,
                p_PScm_E=np.zeros(3),
                G_SP_E=UnitInertia(0.0, 0.0, 0.0),
            ),
        )

        # Define the length and thickness of the arrows for each axis
        axis_length = self.config.triad_scale
        axis_radius = self.config.triad_scale * 0.1

        plant.RegisterVisualGeometry(
            triad_body,
            RigidTransform(p=[axis_length / 2, 0, 0]),
            Box(axis_length, axis_radius, axis_radius),
            f"{name}_x_axis",
            np.array(self.config.RED),
        )
        plant.RegisterVisualGeometry(
            triad_body,
            RigidTransform(p=[0, axis_length / 2, 0]),
            Box(axis_radius, axis_length, axis_radius),
            f"{name}_y_axis",
            np.array(self.config.GREEN),
        )
        plant.RegisterVisualGeometry(
            triad_body,
            RigidTransform(p=[0.0, 0.0, axis_length / 2]),
            Box(axis_radius, axis_radius, axis_length),
            f"{name}_axis",
            np.array(self.config.BLUE),
        )

        return triad_body

    def __add_wall(
        self,
        robot_diagram_builder: RobotDiagramBuilder,
        x: float,
        y: float,
        z: float,
        x_size: float,
        y_size: float,
        z_size: float,
        name: str,
    ):
        plant = robot_diagram_builder.plant()
        parser = robot_diagram_builder.parser()
        parser.SetAutoRenaming(True)
        world_frame = plant.world_frame()

        wall_urdf_path = airo_models.box_urdf_path((x_size, y_size, z_size), name)
        wall_transform = RigidTransform(p=np.array([x, y, z]))
        wall_index = parser.AddModels(wall_urdf_path)[0]
        wall_frame = plant.GetFrameByName("base_link", wall_index)

        plant.WeldFrames(world_frame, wall_frame, wall_transform)
        return wall_index

    def update_spheres(self, spheres: CuroboCollisionSpheresSample):
        context = self.plant.GetMyMutableContextFromRoot(self.context)

        try:
            for i, sphere_position in enumerate(spheres.positions):
                sphere_body = self.sphere_indices[i]
                sphere_transform = RigidTransform(p=sphere_position)
                self.plant.SetFreeBodyPose(context, sphere_body, sphere_transform)

            self.diagram.ForcedPublish(self.context)
        except AttributeError:
            pass



    @property
    def joint_state(self) -> (np.ndarray, np.ndarray):
        context = self.plant.GetMyContextFromRoot(self.context)
        position = self.plant.GetPositions(context, self.drake_indices["robot"]["arm"])
        velocity = self.plant.GetVelocities(context, self.drake_indices["robot"]["arm"])
        return position, velocity

    @joint_state.setter
    def joint_state(self, value: (np.ndarray, np.ndarray)):
        position, velocity = value

        context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.plant.SetPositions(context, self.drake_indices["robot"]["arm"], position)
        self.plant.SetVelocities(context, self.drake_indices["robot"]["arm"], velocity)
        self.diagram.ForcedPublish(self.context)

    @property
    def world_config(self) -> DrakeWorldConfig:
        world_state = DrakeWorldConfig()

        context = self.plant.GetMyContextFromRoot(self.context)
        inspector = self.scene_graph.model_inspector()

        for model_name, model_index in self.drake_indices["world"].items():
            for model_body_index in self.plant.GetBodyIndices(model_index):
                model_body = self.plant.get_body(model_body_index)
                T_world_body = self.plant.EvalBodyPoseInWorld(context, model_body)

                for geometry_id in self.plant.GetCollisionGeometriesForBody(model_body):
                    geometry_shape = inspector.GetShape(geometry_id)
                    T_body_geometry = inspector.GetPoseInFrame(geometry_id)

                    if isinstance(geometry_shape, Box):
                        T_world_body = T_world_body @ T_body_geometry
                        geometry_identifier = (
                            f"{model_name}_{int(model_body_index)}_"
                            f"{geometry_id.get_value()}"
                        )
                        world_state.add_cuboid(
                            geometry_shape, geometry_identifier, T_world_body
                        )
                    elif isinstance(geometry_shape, DrakeSphere):
                        raise NotImplemented

        return world_state

    @property
    def tcp_pose(self) -> np.ndarray:
        return self.__get_triad_pose(self.__ee_triad)

    @tcp_pose.setter
    def tcp_pose(self, value: np.ndarray):
        self.__set_triad_pose(self.__ee_triad, value)

    @property
    def goal_pose(self) -> np.ndarray:
        return self.__get_triad_pose(self.__goal_triad)

    @goal_pose.setter
    def goal_pose(self, value: np.ndarray):
        self.__set_triad_pose(self.__goal_triad, value)

    def __get_triad_pose(self, triad: RigidBody) -> np.ndarray:
        context = self.plant.GetMyMutableContextFromRoot(self.context)
        triad_transform = self.plant.GetFreeBodyPose(context, triad)

        # Convert to 4x4 homogenous matrix
        pose = np.eye(4)
        pose[:3, 3] = triad_transform.translation()
        pose[:3, :3] = triad_transform.rotation().matrix()

        return pose

    def __set_triad_pose(self, triad: RigidBody, homogenous_matrix: np.ndarray) -> None:
        position, quaternion = matrix2cupose(homogenous_matrix, gpu=False)

        context = self.plant.GetMyMutableContextFromRoot(self.context)
        triad_transform = RigidTransform(
            quaternion=Quaternion(quaternion),
            p=position,
        )

        self.plant.SetFreeBodyPose(context, triad, triad_transform)
        self.diagram.ForcedPublish(self.context)


if __name__ == "__main__":
    node = DrakeScene()
