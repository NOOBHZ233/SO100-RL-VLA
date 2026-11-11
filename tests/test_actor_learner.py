"""Exercise real CLI processes, gRPC transport, final partial-episode flush and exit."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_actor_finishes_and_learner_receives_short_rollout(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    cfg = json.loads((ROOT / "config/train_so100_sim.json").read_text())
    cfg.update(batch_size=2, log_freq=1, save_checkpoint=False)
    cfg["policy"].update(
        device="cpu", storage_device="cpu", online_steps=8,
        online_step_before_learning=2, online_buffer_capacity=16, utd_ratio=1,
    )
    cfg["policy"]["actor_learner_config"].update(learner_port=port, policy_parameters_push_frequency=1)
    config = tmp_path / "config.json"
    cfg["output_dir"] = str(tmp_path / "learner-output")
    config.write_text(json.dumps(cfg))
    actor_config = tmp_path / "actor-config.json"
    cfg["output_dir"] = str(tmp_path / "actor-output")
    actor_config.write_text(json.dumps(cfg))
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    processes = []
    logs = {}
    try:
        for role, config_path in (("learner", config), ("actor", actor_config)):
            log_path = tmp_path / f"{role}.log"
            logs[role] = log_path
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", f"project.rl.{role}", "--config_path", str(config_path)],
                    cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            processes.append(process)
        actor = processes[1]
        try:
            actor.wait(timeout=50)
        except subprocess.TimeoutExpired:
            pytest.fail("Actor failed to finish a finite rollout:\n" + logs["actor"].read_text())
        assert actor.returncode == 0, logs["actor"].read_text()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if "Number of optimization step: 2" in logs["learner"].read_text():
                break
            if processes[0].poll() is not None:
                break
            time.sleep(0.2)
        assert "Number of optimization step: 2" in logs["learner"].read_text(), logs["learner"].read_text()
        assert "Traceback" not in logs["actor"].read_text()
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
