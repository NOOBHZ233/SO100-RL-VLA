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
# Gym SO100 - SO100 机械臂 Gym 环境
# ================================================================
# 提供 SO100 6自由度机械臂的 MuJoCo 仿真环境和 Gymnasium 接口
# 支持操作空间控制、人工介入训练、夹爪控制等功能
# ================================================================

import gymnasium as gym

from gym_so100.envs import SO100PickCubeGymEnv
from gym_so100.mujoco_gym_env import SO100GymEnv, GymRenderingSpec
from gym_so100.wrappers import (
    EEActionWrapper,
    GripperPenaltyWrapper,
    InputController,
    InputsControlWrapper,
    KeyboardController,
    PassiveViewerWrapper,
    ResetDelayWrapper,
    make_base_so100_env,
    make_so100_env_for_rollout,
    wrap_env,
)

__all__ = [
    # 环境类
    "SO100GymEnv",
    "SO100PickCubeGymEnv",
    # 渲染配置
    "GymRenderingSpec",
    # 包装器
    "EEActionWrapper",
    "GripperPenaltyWrapper",
    "InputsControlWrapper",
    "ResetDelayWrapper",
    "PassiveViewerWrapper",
    # 控制器
    "InputController",
    "KeyboardController",
    # 工厂函数
    "wrap_env",
    "make_base_so100_env",
    "make_so100_env_for_rollout",
]

__version__ = "0.1.0"

# ================================================================
# Gymnasium 环境注册
# ================================================================

from gymnasium.envs.registration import register

# 注册 SO100 抓取方块基础环境
register(
    id="gym_so100/SO100PickCubeBase-v0",
    entry_point="gym_so100.envs.so100_pick_env:SO100PickCubeGymEnv",
    max_episode_steps=100,
)

# 注册带被动查看器的版本
register(
    id="gym_so100/SO100PickCubeViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(
        gym.make("gym_so100/SO100PickCubeBase-v0", **kwargs)
    ),
    max_episode_steps=100,
)

# 注册标准包装版本
register(
    id="gym_so100/SO100PickCube-v0",
    entry_point="gym_so100.wrappers.factory:make_base_so100_env",
    max_episode_steps=100,
    kwargs={
        "task_name": "pick_cube",
        "use_ee_action": True,
        "use_gripper": True,
        "gripper_penalty": -0.05,
    },
)

# 注册手柄控制版本
register(
    id="gym_so100/SO100PickCubeGamepad-v0",
    entry_point="gym_so100.wrappers.factory:make_base_so100_env",
    max_episode_steps=100,
    kwargs={
        "task_name": "pick_cube",
        "use_ee_action": True,
        "use_gripper": True,
        "passive_viewer": True,
        "use_input_control": True,
        "use_gamepad": True,
        "auto_reset": True,
        "reset_delay": 1.0,
    },
)

# 注册键盘控制版本
register(
    id="gym_so100/SO100PickCubeKeyboard-v0",
    entry_point="gym_so100.wrappers.factory:make_base_so100_env",
    max_episode_steps=100,  # 增加步数限制，避免过早结束
    kwargs={
        "task_name": "pick_cube",
        "render_mode": "human",  # 使用human模式以支持PassiveViewerWrapper
        "image_obs": True,  # 启用图像观察
        "use_ee_action": True,
        "use_gripper": True,
        "gripper_penalty": -0.05,
        "passive_viewer": True,
        "use_input_control": True,
        "use_gamepad": False,
        "require_intervention_key": False,  # 不需要按空格键,直接键盘控制
        "auto_reset": True,  # 禁用自动重置，避免意外关闭
        "reset_delay": 0.0,  # 无延迟
    },
)

# 注册用于数据采集的遥操作环境
register(
    id="gym_so100/SO100PickCubeRollout-v0",
    entry_point="gym_so100.wrappers.factory:make_so100_env_for_rollout",
    max_episode_steps=100,
    kwargs={
        "task_name": "pick_cube",
        "use_gamepad": True,
        "passive_viewer": True,
        "auto_reset": True,
        "reset_delay": 1.0,
    },
)
