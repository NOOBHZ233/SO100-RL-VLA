"""Offline integration check: cameras → processors → SAC → replay → gradient update.

Run from a checkout with:
    MUJOCO_GL=osmesa python -m project.rl.smoke --config_path config/train_so100_sim.json
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
import torch

from project.configs.train import TrainRLServerPipelineConfig
from project.policies.factory import make_policy
from project.processor import TransitionKey, create_transition
from project.rl.buffer import ReplayBuffer
from project.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from project.transport import services_pb2, services_pb2_grpc  # noqa: F401
from project.utils.random_utils import set_seed


def run_smoke(config_path: str | Path, device: str = "cpu", steps: int = 3) -> dict:
    if steps < 2:
        raise ValueError("At least two steps are required for replay sampling.")
    torch.set_num_threads(2)
    with draccus.config_type("json"):
        cfg = draccus.parse(TrainRLServerPipelineConfig, str(config_path), args=[])
    cfg.policy.device = device
    cfg.policy.storage_device = "cpu"
    # Fail if the documented offline baseline starts requiring remote assets.
    assert cfg.dataset is None, "Smoke config must use online data only"
    assert cfg.policy.vision_encoder_name is None, "Smoke config must use the local encoder"
    assert not cfg.wandb.enable and not cfg.policy.push_to_hub
    cfg.validate()
    set_seed(cfg.seed)

    env, teleop = make_robot_env(cfg.env)
    try:
        obs, info = env.reset(seed=cfg.seed)
        assert env.observation_space.contains(obs)
        camera_shapes = {key: list(value.shape) for key, value in obs["pixels"].items()}
        env_processor, action_processor = make_processors(env, teleop, cfg.env, device)
        transition = env_processor(create_transition(observation=obs, info=info))
        policy = make_policy(cfg.policy, env_cfg=cfg.env)
        policy.eval()
        buffer = ReplayBuffer(
            capacity=steps + 1, device=device, storage_device="cpu",
            state_keys=cfg.policy.input_features.keys(), use_drq=False,
        )
        for _ in range(steps):
            state = {key: transition[TransitionKey.OBSERVATION][key] for key in cfg.policy.input_features}
            action = policy.select_action(state)
            assert tuple(action.shape) == (1, 4), action.shape
            assert torch.isfinite(action).all()
            next_transition = step_env_and_process_transition(
                env, transition, action, env_processor, action_processor,
            )
            next_state = {
                key: next_transition[TransitionKey.OBSERVATION][key]
                for key in cfg.policy.input_features
            }
            assert all(torch.isfinite(value).all() for value in next_state.values())
            buffer.add(
                state, action, float(next_transition[TransitionKey.REWARD]), next_state,
                bool(next_transition[TransitionKey.DONE]), bool(next_transition[TransitionKey.TRUNCATED]),
            )
            transition = next_transition
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                env_processor.reset()
                action_processor.reset()
                obs, info = env.reset()
                transition = env_processor(create_transition(observation=obs, info=info))

        batch = buffer.sample(batch_size=2)
        policy.train()
        parameters = policy.get_optim_params()
        losses = {}
        for name in ("critic", "actor", "temperature"):
            group = parameters[name]
            if isinstance(group, torch.Tensor):
                group = [group]
            optimizer = torch.optim.Adam(group, lr=3e-4)
            policy.zero_grad(set_to_none=True)
            loss = next(iter(policy(batch, model=name).values()))
            assert torch.isfinite(loss), (name, loss)
            loss.backward()
            assert any(p.grad is not None for p in optimizer.param_groups[0]["params"])
            optimizer.step()
            losses[name] = float(loss.detach())
        policy.update_target_networks()

        # Generated protobuf must import, serialize, and round-trip with our runtime.
        payload = services_pb2.Parameters(data=b"so100-smoke")
        assert services_pb2.Parameters.FromString(payload.SerializeToString()).data == b"so100-smoke"
        with TemporaryDirectory() as directory:
            policy.save_pretrained(directory)
            restored = type(policy).from_pretrained(directory, local_files_only=True)
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(value.cpu(), restored.state_dict()[key].cpu())
        return {
            "status": "ok", "device": device, "steps": steps,
            "cameras": camera_shapes, "action_shape": [1, 4], "losses": losses,
            "checkpoint_roundtrip": True, "protobuf_roundtrip": True,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, default=Path("config/train_so100_sim.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config_path, args.device, args.steps), indent=2))


if __name__ == "__main__":
    main()
