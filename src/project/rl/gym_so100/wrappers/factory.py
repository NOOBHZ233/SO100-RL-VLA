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
    from gym_so100.wrappers.hil_wrappers import (
        EEActionWrapper,
        GripperPenaltyWrapper,
        InputsControlWrapper,
        ResetDelayWrapper,
    )
    from gym_so100.wrappers.viewer_wrapper import PassiveViewerWrapper

    if passive_viewer:
        env = PassiveViewerWrapper(env)

    if use_ee_action:
        env = EEActionWrapper(env, ee_action_step_size=ee_action_step_size, use_gripper=use_gripper)

    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)

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

    import gym_so100

    if task_name == "pick_cube":
        env = gym.make(
            "gym_so100/SO100PickCubeBase-v0",
            render_mode=render_mode,
            image_obs=image_obs,
            seed=seed,
        )
    elif task_name == "reach":
        raise NotImplementedError(f"Task '{task_name}' not yet implemented")
    else:
        raise ValueError(f"Unknown task name: {task_name}")

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
