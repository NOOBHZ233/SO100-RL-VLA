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
# 环境工厂函数
# ================================================================
# 提供包装环境和创建环境的便捷函数
# 适用于 SO100 机械臂 Gym 环境
# ================================================================

import logging
from typing import Literal, Optional

import gymnasium as gym


def wrap_env(
    env: gym.Env,
    *,
    image_obs: bool = False,
    reward_scale: float = 1.0,
    terminate_on_collision: bool = False,
    terminate_on_success: bool = True,
    episode_length: int = 200,
    ee_action_step_size: dict = {"x": 0.25, "y": 0.25, "z": 0.25},
    use_ee_action: bool = True,
    use_gripper: bool = True,
    gripper_penalty: float = -0.05,
    input_threshold: float = 0.001,
    use_gamepad: bool = True,
    controller_config_path: Optional[str] = None,
    use_input_control: bool = False,
    require_intervention_key: bool = False,
    auto_reset: bool = False,
    reset_delay: float = 0.0,
    passive_viewer: bool = False,
    from_pixels: bool = False,
) -> gym.Wrapper:
    """包装 SO100 环境并应用一系列包装器

    在不修改原始环境底层代码的前提下，为环境动态添加新功能或修改其输入输出。ZCW


    Args:
        env: 要包装的基础 Gym 环境
        image_obs: 是否使用图像观察
        reward_scale: 奖励缩放因子
        terminate_on_collision: 碰撞时是否终止
        terminate_on_success: 成功时是否终止
        episode_length: 最大 episode 长度
        ee_action_step_size: 末端执行器动作步长 (米)
        use_ee_action: 是否使用末端执行器动作空间
        use_gripper: 是否使用夹爪控制
        gripper_penalty: 夹爪惩罚值
        input_threshold: 人工输入阈值
        use_gamepad: 使用手柄还是键盘
        controller_config_path: 控制器配置路径
        use_input_control: 是否启用人工输入控制
        require_intervention_key: 键盘模式是否需要空格键启用干预
        auto_reset: episode 结束时是否自动重置
        reset_delay: 重置延迟 (秒)
        passive_viewer: 是否启用被动查看器
        from_pixels: 是否从像素创建观察

    Returns:
        包装后的 Gym 环境
    """
    from gym_so100.wrappers.hil_wrappers import (
        EEActionWrapper,
        GripperPenaltyWrapper,
        InputsControlWrapper,
        ResetDelayWrapper,
    )
    from gym_so100.wrappers.viewer_wrapper import PassiveViewerWrapper

    # 1. 被动查看器包装器（如果启用）
    if passive_viewer:
        env = PassiveViewerWrapper(env)

    # 2. 末端执行器动作空间包装器
    if use_ee_action:
        env = EEActionWrapper(env, ee_action_step_size=ee_action_step_size, use_gripper=use_gripper)

    # 3. 夹爪惩罚包装器
    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)

    # 4. 人工输入控制包装器
    if use_input_control:
        env = InputsControlWrapper(
            env,
            x_step_size=ee_action_step_size["x"],
            y_step_size=ee_action_step_size["y"],
            z_step_size=ee_action_step_size["z"],
            use_gripper=use_gripper,
            auto_reset=auto_reset,
            input_threshold=input_threshold,
            use_gamepad=use_gamepad,
            controller_config_path=controller_config_path,
            require_intervention_key=require_intervention_key,
        )

    # 5. 重置延迟包装器
    if reset_delay > 0:
        env = ResetDelayWrapper(env, delay_seconds=reset_delay)

    return env


def make_base_so100_env(
    task_name: str,
    *,
    image_obs: bool = False,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    reward_scale: float = 1.0,
    terminate_on_collision: bool = False,
    terminate_on_success: bool = True,
    episode_length: int = 200,
    ee_action_step_size: dict = {"x": 0.25, "y": 0.25, "z": 0.25},
    use_ee_action: bool = True,
    use_gripper: bool = True,
    gripper_penalty: float = -0.05,
    input_threshold: float = 0.001,
    use_gamepad: bool = True,
    controller_config_path: Optional[str] = None,
    use_input_control: bool = False,
    require_intervention_key: bool = False,
    auto_reset: bool = False,
    reset_delay: float = 0.0,
    passive_viewer: bool = False,
    from_pixels: bool = False,
    width: int = 128,
    height: int = 128,
    seed: int = 0,
) -> gym.Env:
    """创建 SO100 Gym 环境的便捷工厂函数

    Args:
        task_name: 任务名称 ("pick_cube", "reach", 等)
        image_obs: 是否使用图像观察
        render_mode: 渲染模式 ("rgb_array" 或 "human")
        reward_scale: 奖励缩放因子
        terminate_on_collision: 碰撞时是否终止
        terminate_on_success: 成功时是否终止
        episode_length: 最大 episode 长度
        ee_action_step_size: 末端执行器动作步长 (米)
        use_ee_action: 是否使用末端执行器动作空间
        use_gripper: 是否使用夹爪控制
        gripper_penalty: 夹爪惩罚值
        input_threshold: 人工输入阈值
        use_gamepad: 使用手柄还是键盘
        controller_config_path: 控制器配置路径
        use_input_control: 是否启用人工输入控制
        require_intervention_key: 键盘模式是否需要空格键启用干预
        auto_reset: episode 结束时是否自动重置
        reset_delay: 重置延迟 (秒)
        passive_viewer: 是否启用被动查看器
        from_pixels: 是否从像素创建观察
        width: 图像宽度
        height: 图像高度
        seed: 随机种子

    Returns:
        配置好的 Gym 环境
    """
    import gym_so100

    # 根据任务名称创建环境
    if task_name == "pick_cube":
        env = gym.make(
            "gym_so100/SO100PickCubeBase-v0",
            render_mode=render_mode,
            image_obs=image_obs,
            seed=seed,
        )
    elif task_name == "reach":
        # 可以添加更多任务类型
        raise NotImplementedError(f"Task '{task_name}' not yet implemented")
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    # 应用包装器
    env = wrap_env(
        env,
        image_obs=image_obs,
        reward_scale=reward_scale,
        terminate_on_collision=terminate_on_collision,
        terminate_on_success=terminate_on_success,
        episode_length=episode_length,
        ee_action_step_size=ee_action_step_size,
        use_ee_action=use_ee_action,
        use_gripper=use_gripper,
        gripper_penalty=gripper_penalty,
        input_threshold=input_threshold,
        use_gamepad=use_gamepad,
        controller_config_path=controller_config_path,
        use_input_control=use_input_control,
        require_intervention_key=require_intervention_key,
        auto_reset=auto_reset,
        reset_delay=reset_delay,
        passive_viewer=passive_viewer,
        from_pixels=from_pixels,
    )

    return env


def make_so100_env_for_rollout(
    task_name: str = "pick_cube",
    *,
    ee_action_step_size: dict = {"x": 0.025, "y": 0.025, "z": 0.025},
    use_gripper: bool = True,
    use_gamepad: bool = True,
    controller_config_path: Optional[str] = None,
    require_intervention_key: bool = False,
    auto_reset: bool = True,
    reset_delay: float = 1.0,
    passive_viewer: bool = False,
    width: int = 128,
    height: int = 128,
    seed: int = 0,
) -> gym.Env:
    """为遥操作数据采集创建 SO100 环境

    专门用于人工收集演示数据的环境配置。

    Args:
        task_name: 任务名称
        ee_action_step_size: 末端执行器动作步长
        use_gripper: 是否使用夹爪
        use_gamepad: 使用手柄还是键盘
        controller_config_path: 控制器配置路径
        require_intervention_key: 键盘模式是否需要空格键启用干预
        auto_reset: 自动重置
        reset_delay: 重置延迟
        passive_viewer: 是否显示被动查看器
        width: 图像宽度
        height: 图像高度
        seed: 随机种子

    Returns:
        配置好的环境
    """
    return make_base_so100_env(
        task_name=task_name,
        render_mode="rgb_array",
        image_obs=True,
        ee_action_step_size=ee_action_step_size,
        use_ee_action=True,
        use_gripper=use_gripper,
        input_threshold=0.001,
        use_gamepad=use_gamepad,
        controller_config_path=controller_config_path,
        use_input_control=True,
        require_intervention_key=require_intervention_key,
        auto_reset=auto_reset,
        reset_delay=reset_delay,
        passive_viewer=passive_viewer,
        width=width,
        height=height,
        seed=seed,
    )
