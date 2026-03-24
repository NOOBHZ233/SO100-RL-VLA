#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================================
# MuJoCo Robot Arm Gym Environment Base Class
# ================================================================
# Provides Gymnasium environment interface for 6DOF robots like SO100
# Contains core functionality for operational space control, observation space, action space, etc.
# ================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_so100.controllers import opspace

# Maximum gripper control command value (corresponds to fully closed)
MAX_GRIPPER_COMMAND = 255

# Class cannot be modified after creation
@dataclass(frozen=True)
class GymRenderingSpec:
    """Rendering configuration parameters

    Attributes:
        height: Image height (pixels)
        width: Image width (pixels)
        camera_id: Camera ID (-1 for default camera)
        mode: Rendering mode ("rgb_array" or "human")
    """
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MuJoCo Environment Gym Interface Base Class

    Provides a bridge between MuJoCo physics simulation and Gymnasium standard interface.
    Handles core functionality like model loading, rendering, physics stepping, etc.
    """

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        """Initialize MuJoCo Environment

        Args:
            xml_path: Path to MuJoCo XML file
            seed: Random seed
            control_dt: Control period (seconds), i.e., interval between each action
            physics_dt: Physics simulation timestep (seconds), MuJoCo internal timestep
            render_spec: Rendering configuration
        """
        # Load MuJoCo model from XML
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        # Set offscreen rendering dimensions
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        # Create simulation data
        self._data = mujoco.MjData(self._model)
        # Set physics simulation timestep
        self._model.opt.timestep = physics_dt
        # Control period
        self._control_dt = control_dt
        # Number of physics substeps per control period
        self._n_substeps = int(control_dt // physics_dt)
        # Random number generator
        self._random = np.random.RandomState(seed)
        # Renderer (lazy initialization)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            # For human mode, the viewer is managed by external wrapper
            return None

        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        if hasattr(viewer, "close") and callable(viewer.close):
            try:  # noqa: SIM105
                viewer.close()
            except Exception:
        
                pass

        self._viewer = None


    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random


class SO100GymEnv(MujocoGymEnv):

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(), 
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0.0, -0.8, 1.5, 0.0, 0.0, 0.0)),  
        cartesian_bounds: np.ndarray = np.asarray([[-0.4, -0.4,  -0.4], [0.4, 0.4, 0.4]]),  
    ):

        if xml_path is None:
            xml_path = Path(__file__).parent / "model" / "scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs

        camera_name_1 = "front"
        camera_name_2 = "gripper_cam"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)

        so100_arm_joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self._so100_arm_dof_ids = np.asarray([self._model.joint(name).id for name in so100_arm_joint_names])
        so100_arm_actuator_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self._so100_arm_ctrl_ids = np.asarray([self._model.actuator(name).id for name in so100_arm_actuator_names])
        self._gripper_joint_name = "Jaw"
        self._gripper_dof_id = self._model.joint(self._gripper_joint_name).id
        self._gripper_ctrl_id = self._model.actuator("Jaw").id
        try:
            self._gripper_site_id = self._model.site("gripper_site").id
        except Exception:
            
            print("Warning: gripper_site not found, using default site")
            if self._model.nsite > 0:
                self._gripper_site_id = 0
            else:
                self._gripper_site_id = -1

        self._setup_observation_space()
        self._setup_action_space()

        if self.render_mode == "rgb_array":
            self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        else:
            self._viewer = None

    def _setup_observation_space(self):
        base_obs = {
            "joint_pose": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        }

        if self.image_obs:
            base_obs["pixels"] = spaces.Dict(
                {
                    "front": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                }
            )

        self.observation_space = spaces.Dict(base_obs)

    def _setup_action_space(self):
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        # Set joint positions (first 5 joints + gripper)
        self._data.qpos[self._so100_arm_dof_ids] = self._home_position[:5]
        self._data.qpos[self._gripper_dof_id] = self._home_position[5]
        # Zero joint velocities
        self._data.qvel[self._so100_arm_dof_ids] = 0.0
        self._data.qvel[self._gripper_dof_id] = 0.0
        # Zero joint control torques
        self._data.ctrl[self._so100_arm_ctrl_ids] = 0.0
        self._data.ctrl[self._gripper_ctrl_id] = 0.0
        # Forward kinematics update
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action):
        if len(action) == 4:
            x, y, z, grasp_command = action
        else:
            print(action)
            raise ValueError("dim of action is INVALID")
        current_pos = self._data.sensor("so100/gripper_site_pos").data
        dpos = np.asarray([x, y, z])
        # print(dpos)
        # print(current_pos)
        target_pos = np.clip(current_pos + dpos, *self._cartesian_bounds)
        # print(target_pos)
        gripper_range = [-0.2, 2.0]
        target_gripper_pos = ((grasp_command + 1.0) / 2.0) * (gripper_range[1] - gripper_range[0]) + gripper_range[0]
        target_gripper_pos = np.clip(target_gripper_pos, *gripper_range)

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._gripper_site_id,
                dof_ids=self._so100_arm_dof_ids,
                pos=target_pos,
                ori=None,
                joint=self._home_position[:5],
                gravity_comp=True,
                nullspace_stiffness=2,
                pos_gains=(300.0, 300.0, 300.0),
            )
            self._data.ctrl[self._so100_arm_ctrl_ids] = tau
            # print(tau)
            kp_gripper = 8.0
            kd_gripper = 2.0
            current_gripper_pos = self._data.qpos[self._gripper_dof_id]
            current_gripper_vel = self._data.qvel[self._gripper_dof_id]
            gripper_tau = kp_gripper * (target_gripper_pos - current_gripper_pos) - kd_gripper * current_gripper_vel
            gripper_tau = np.clip(gripper_tau, -20.0, 20.0)
            self._data.ctrl[self._gripper_ctrl_id] = gripper_tau

            mujoco.mj_step(self._model, self._data)

    def get_joint_pose(self):
        return self.data.qpos[self._so100_arm_dof_ids].astype(np.float32)

    def get_robot_state(self):
        arm_qpos = self.data.qpos[self._so100_arm_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()
        gripper_pose = np.clip(gripper_pose, 0.0, 1.0)
        return np.concatenate([arm_qpos, gripper_pose])

    def render(self):
        
        if self._viewer is None:
            # Viewer is managed by PassiveViewerWrapper in human mode
            return []

        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames

    def get_gripper_pose(self):

        gripper_joint_pos = self._data.qpos[self._gripper_dof_id]
        gripper_min = -0.2
        gripper_max = 2.0
        gripper_normalized = (gripper_joint_pos - gripper_min) / (gripper_max - gripper_min)
        gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)
        return np.array([gripper_normalized], dtype=np.float32)
