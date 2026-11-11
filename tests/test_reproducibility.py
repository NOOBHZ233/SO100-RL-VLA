"""Offline regression checks for the documented SO100 baseline."""

import importlib
import json
from pathlib import Path

import draccus
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from project.configs.train import TrainRLServerPipelineConfig
from project.rl.gym_manipulator import GymManipulatorConfig
from project.rl.gym_so100.envs.so100_pick_env import SO100PickCubeGymEnv
from project.rl.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_offline_sac_pipeline():
    report = run_smoke(ROOT / "config/train_so100_sim.json")
    assert report["status"] == "ok"
    assert report["cameras"] == {"front": [128, 128, 3], "wrist": [128, 128, 3]}
    assert report["checkpoint_roundtrip"] and report["protobuf_roundtrip"]


def test_seeded_environment_matches_gymnasium_contract():
    env = SO100PickCubeGymEnv(render_mode="human", random_block_position=True)
    try:
        check_env(env, skip_render_check=True)
        env.reset(seed=23)
        first_position = env.data.jnt("block").qpos.copy()
        env.reset(seed=99)
        different_position = env.data.jnt("block").qpos.copy()
        env.reset(seed=23)
        np.testing.assert_array_equal(first_position, env.data.jnt("block").qpos)
        assert not np.array_equal(first_position, different_position)
    finally:
        env.close()


def test_registered_environment_respects_camera_size():
    env = gym.make("gym_so100/SO100PickCube-v0", image_obs=True, width=96, height=80)
    try:
        observation, _ = env.reset(seed=12)
        assert env.observation_space.contains(observation)
        assert observation["pixels"]["front"].shape == (80, 96, 3)
        assert observation["pixels"]["wrist"].shape == (80, 96, 3)
        assert env.action_space.shape == (4,)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    finally:
        env.close()


@pytest.mark.parametrize("filename,config_type", [
    ("train_so100_sim.json", TrainRLServerPipelineConfig),
    ("manual_so100.json", GymManipulatorConfig),
])
def test_shipped_configs_parse_without_network(filename, config_type):
    with draccus.config_type("json"):
        cfg = draccus.parse(config_type, str(ROOT / "config" / filename), args=[])
    if isinstance(cfg, TrainRLServerPipelineConfig):
        cfg.validate()
        assert cfg.dataset is None
        assert not cfg.wandb.enable and not cfg.policy.push_to_hub
        assert cfg.env.features["observation.state"].shape == (6,)
        assert cfg.policy.output_features["action"].shape == (4,)
        assert cfg.policy.num_discrete_actions is None
    else:
        assert cfg.mode is None
        assert cfg.device == "cpu" and not cfg.dataset.push_to_hub


def test_optional_policy_configs_do_not_load_models():
    # Merely selecting SAC must work without diffusers, peft or flash-attn.
    import sys
    for name in ("groot", "pi0", "pi05"):
        importlib.import_module(f"project.policies.{name}.configuration_{name}")
        assert f"project.policies.{name}.modeling_{name}" not in sys.modules


def test_intervention_action_is_saved_in_replay(monkeypatch):
    import torch
    from project.processor import TransitionKey, create_transition
    from project.rl.gym_manipulator import make_processors, make_robot_env, step_env_and_process_transition

    with draccus.config_type("json"):
        cfg = draccus.parse(TrainRLServerPipelineConfig, str(ROOT / "config/train_so100_sim.json"), args=[])
    env, teleop = make_robot_env(cfg.env)
    actual_action = np.asarray([0.1, -0.2, 0.0, -1.0], dtype=np.float32)
    original_step = env.step

    def intervened_step(action):
        obs, reward, terminated, truncated, info = original_step(actual_action)
        info["teleop_action"] = actual_action
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(env, "step", intervened_step)
    try:
        env_processor, action_processor = make_processors(env, teleop, cfg.env)
        obs, info = env.reset(seed=1)
        transition = env_processor(create_transition(observation=obs, info=info))
        result = step_env_and_process_transition(
            env, transition, torch.zeros(1, 4), env_processor, action_processor,
        )
        torch.testing.assert_close(
            result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"],
            torch.from_numpy(actual_action).unsqueeze(0),
        )
    finally:
        env.close()


def test_recorded_dataset_can_be_loaded_and_replayed(tmp_path, monkeypatch):
    from project.datasets.lerobot_dataset import LeRobotDataset
    from project.rl import gym_manipulator as manipulator

    with draccus.config_type("json"):
        cfg = draccus.parse(GymManipulatorConfig, str(ROOT / "config/manual_so100.json"), args=[])
    cfg.env.task = "SO100PickCube-v0"
    cfg.mode = "record"
    cfg.dataset.num_episodes_to_record = 1
    cfg.dataset.root = str(tmp_path / "dataset")
    cfg.dataset.replay_episode = 0
    env, teleop = manipulator.make_robot_env(cfg.env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3)
    env_processor, action_processor = manipulator.make_processors(env, teleop, cfg.env)
    monkeypatch.setattr(manipulator, "precise_sleep", lambda duration: None)
    try:
        manipulator.control_loop(env, env_processor, action_processor, teleop, cfg)
        dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, video_backend="pyav")
        assert dataset.num_episodes == 1 and dataset.num_frames == 3
        assert dataset[0]["action"].shape == (4,)
        assert dataset[0]["observation.images.front"].shape == (3, 128, 128)
        manipulator.replay_trajectory(env, action_processor, cfg)
    finally:
        env.close()


def test_gripper_penalty_uses_normalized_position():
    from project.rl.gym_so100.wrappers.hil_wrappers import GripperPenaltyWrapper

    class NormalizedGripper(gym.Env):
        def get_gripper_pose(self):
            return np.array([0.95], dtype=np.float32)

        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    env = GripperPenaltyWrapper(NormalizedGripper(), penalty=-0.02)
    env.reset()
    _, _, _, _, info = env.step(np.array([0, 0, 0, -1]))
    assert info["discrete_penalty"] == -0.02
