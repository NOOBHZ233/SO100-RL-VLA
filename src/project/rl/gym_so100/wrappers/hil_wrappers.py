#!/usr/bin/env python

import logging
import sys
import time

import gymnasium as gym
import numpy as np

from gym_so100.mujoco_gym_env import MAX_GRIPPER_COMMAND

DEFAULT_EE_STEP_SIZE = {"x": 0.025, "y": 0.025, "z": 0.025}


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["discrete_penalty"] = 0.0
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.1
        ):
            info["discrete_penalty"] = self.penalty
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info


class EEActionWrapper(gym.Wrapper):
    def __init__(self, env, ee_action_step_size, use_gripper=True):
        super().__init__(env)
        self.ee_action_step_size = ee_action_step_size
        self.use_gripper = use_gripper

        self._ee_step_size = np.array(
            [
                ee_action_step_size["x"],
                ee_action_step_size["y"],
                ee_action_step_size["z"],
            ]
        )

        num_actions = 4 if use_gripper else 3
        action_space_bounds_min = -np.ones(num_actions)
        action_space_bounds_max = np.ones(num_actions)

        ee_action_space = gym.spaces.Box(
            low=action_space_bounds_min,
            high=action_space_bounds_max,
            shape=(num_actions,),
            dtype=np.float32,
        )
        self.action_space = ee_action_space

    def step(self, action):
        grasp_command = 0.0
        if self.use_gripper:
            grasp_command = action[3]
            ee_action = action[:3]
        else:
            ee_action = action

        action_xyz = ee_action * self._ee_step_size
        base_action = np.concatenate([action_xyz, [grasp_command]])
        return self.env.step(base_action)


class InputsControlWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        x_step_size=10,
        y_step_size=10,
        z_step_size=10,
        use_gripper=True,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        controller_config_path=None,
        require_intervention_key=False,
    ):
        super().__init__(env)
        from gym_so100.wrappers.intervention_utils import (
            GamepadController,
            GamepadControllerHID,
            KeyboardController,
        )

        if use_gamepad:
            if sys.platform == "darwin":
                self.controller = GamepadControllerHID(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                )
            else:
                self.controller = GamepadController(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
                require_intervention_key=require_intervention_key,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

    def get_gamepad_action(self):
        self.controller.update()
        delta_x, delta_y, delta_z = self.controller.get_deltas()
        intervention_is_active = self.controller.should_intervene()

        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [1.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [-1.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [0.0]])

        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        if terminate_episode:
            logging.info(f"Manually ending episode: {'success' if success else 'failure'}")

        if is_intervention:
            action = gamepad_action

        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully, reward 1.0")

        info["is_intervention"] = is_intervention
        info["teleop_action"] = action
        info["rerecord_episode"] = rerecord_episode

        if terminated or truncated:
            info["next.success"] = success
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.controller.reset()
        return self.env.reset(**kwargs)

    def close(self):
        if hasattr(self, "controller"):
            self.controller.stop()
        return self.env.close()


class ResetDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay_seconds=1.0):
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        logging.info(f"Reset delay {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)
        return self.env.reset(**kwargs)
