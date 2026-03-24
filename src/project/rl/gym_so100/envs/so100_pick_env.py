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
# SO100 Capture Block Task Environment
# ================================================================

from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_so100.mujoco_gym_env import SO100GymEnv, GymRenderingSpec

_SO100_HOME = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

_CARTESIAN_BOUNDS = np.asarray([[-0.2,-0.35,0.027],[0.2,-0.15,0.38]])

_SAMPLING_BOUNDS = np.asarray([[-0.5, -0.5], [0.5, 0.5]])


class SO100PickCubeGymEnv(SO100GymEnv):
    """SO100 机械臂抓取方块任务环境

    任务描述:
    - SO100 机械臂需要抓取工作空间内的方块
    - 将方块抬升至少 0.1m 即视为成功
    - 支持 sparse 和 dense 两种奖励模式
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
    ):
        self.reward_type = reward_type
        self.render_mode = render_mode

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_SO100_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position

        # self._update_observation_space()

    # def _update_observation_space(self):
    #     from gymnasium import spaces

    #     base_obs = dict(self.observation_space.spaces)
    #     base_obs["block_pos"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    #     self.observation_space = spaces.Dict(base_obs)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset Environment

        Returns:
            observation: Initial observation
            info: Additional Information Dictionary
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        self.reset_robot()

        if self._random_block_position:

            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        else:
            block_xy = np.asarray([-0.02, -0.3])
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.1 

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()
        # print(len(obs))
        if self.reward_type == "sparse":
            success = rew == 1.0

        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        observation = {}

        joint_pose = self.get_joint_pose().astype(np.float32)
        gripper_pose = self.get_gripper_pose().astype(np.float32)
        gripper_pose = np.clip(gripper_pose, 0.0, 1.0)

        observation["joint_pose"] = joint_pose
        observation["gripper_pose"] = gripper_pose

        # block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        # observation["block_pos"] = block_pos

        if self.image_obs:
            rendered_frames = self.render()
            if not rendered_frames and self.render_mode == "human":
                import mujoco
                temp_renderer = mujoco.Renderer(
                    self._model,
                    height=self._render_specs.height,
                    width=self._render_specs.width,
                )
                temp_renderer.update_scene(self._data, camera="front")
                front_view = temp_renderer.render()
                temp_renderer.update_scene(self._data, camera="gripper_cam")
                gripper_view = temp_renderer.render()
                temp_renderer.close()
            else:
                front_view, gripper_view = rendered_frames

            observation["pixels"] = {"front": front_view, "wrist": gripper_view}

        return observation

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data

        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("so100/gripper_site_pos").data

            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)

            return 0.3 * r_close + 0.7 * r_lift
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.1)

    def _is_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("so100/gripper_site_pos").data

        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init

        return dist < 0.05 and lift > 0.1


if __name__ == "__main__":
    from gym_so100.wrappers.viewer_wrapper import PassiveViewerWrapper

    env = SO100PickCubeGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    env.reset()

    for _ in range(100):
        action = np.random.uniform(-1, 1, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("succeed"):
            print("Task succeeded!")
            break

    env.close()
