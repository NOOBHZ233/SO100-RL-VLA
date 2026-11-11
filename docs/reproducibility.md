# Reproducing SO100-RL-VLA

[English home](../README.md) · [中文首页](../README.zh-CN.md)

This guide separates a runnable software baseline from experimental results.
The repository contains the SO100 simulator and training code. It does not yet
contain the trained policies, demonstration data or complete hardware setup
used in the demo GIFs. No task-success rate or training-time guarantee is made.

## Environment

| Layer | Baseline |
| :--- | :--- |
| Platform | Linux x86_64; local checks on Ubuntu 20.04 |
| Python | 3.10.19; package support is limited to Python 3.10 |
| ML runtime | torch 2.7.1 + torchvision 0.22.1; choose CPU or CUDA 12.6 wheels |
| Simulator | Native `mujoco` 3.4.0 + Gymnasium 1.2.3 |
| Transport | grpcio 1.73.1 + protobuf 6.31.1, matching the generated source |
| Data | datasets 4.1.1 + PyArrow 23.0.0 + PyAV 15.1.0 |
| Video | FFmpeg 7.1.1 in the Conda environment; PyAV decoder is the default fallback |
| Python dependencies | [`constraints-py310.txt`](../requirements/constraints-py310.txt) |

Use the installation commands in the [README](../README.md#quick-start). Conda
provides Python, pip and FFmpeg; pip installs the Python stack. The native
[`mujoco` wheel](https://mujoco.readthedocs.io/en/stable/python.html#installation)
includes MuJoCo itself. `mujoco-py`, a separate MuJoCo installation, the legacy
`gym` package, and an installed `lerobot` package are not required for SAC.

Install PyTorch from the [official wheel matrix](https://pytorch.org/get-started/previous-versions/#v2-7-1).
The NVIDIA driver must support the wheel you choose. Installing a system CUDA
toolkit is not part of the baseline. CPU constraints cover the resolved Python
packages; CUDA runtime packages, OS packages and optional extras are not a
complete cross-platform lock.

For a venv instead of Conda, use an existing Python 3.10 interpreter:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.3
# Then install CPU/CUDA torch and the project as shown in README.md.
# Install FFmpeg through your OS package manager if you record/export video.
```

### Rendering

Set the backend **before starting Python**:

```bash
# Headless software rendering (CPU machines and CI)
MUJOCO_GL=osmesa HF_HUB_OFFLINE=1 python -m project.rl.smoke

# Headless NVIDIA rendering
MUJOCO_GL=egl HF_HUB_OFFLINE=1 python -m project.rl.smoke

# Check the CUDA policy as well, after installing CUDA torch
MUJOCO_GL=egl HF_HUB_OFFLINE=1 python -m project.rl.smoke --device cuda
```

OSMesa needs `libosmesa6`; EGL needs an available EGL implementation and, for
NVIDIA, a working driver. The keyboard viewer needs a graphical desktop and
an X11 input backend. EGL/OSMesa does not replace the desktop required by the
interactive viewer. Native Wayland keyboard capture and non-Linux systems
are outside the verified baseline.

## Verification

```bash
python -m pip check
MUJOCO_GL=osmesa HF_HUB_OFFLINE=1 python -m project.rl.smoke
MUJOCO_GL=osmesa HF_HUB_OFFLINE=1 python -m pytest
python -m project.rl.actor --help
python -m project.rl.learner --help
python -m project.rl.gym_manipulator --help
```

The smoke check uses the same environment factory, observation/action processors,
policy factory and replay buffer as training. It checks both RGB cameras at
128 × 128, the six-value joint/gripper state, a four-value action, finite
losses for all three SAC objectives, target updates, checkpoint weight
round-tripping and the generated Protobuf runtime. The tests additionally
check seeded resets, Gymnasium's environment contract, custom camera sizes,
configuration parsing, optional-model import isolation, real CLI/gRPC transport,
short-rollout shutdown, and local dataset recording/replay.

The [GitHub Actions template](reproducibility-workflow.yml) can be enabled by
copying it to `.github/workflows/reproducibility.yml` using GitHub or credentials
that permit workflow changes. Once enabled, it installs the CPU stack on Ubuntu
22.04, runs these checks, builds a wheel and sdist, then installs the wheel and
runs the smoke check from outside the checkout. This catches missing MJCF/mesh
files and accidental dependence on the local source tree. Local verification
details are in [validation.md](validation.md).

## Training and recording

### Online SAC

Activate the environment in each terminal. Start the learner before the actor:

```bash
# Terminal 1
python -m project.rl.learner --config_path config/train_so100_sim.json
```

```bash
# Terminal 2
MUJOCO_GL=egl python -m project.rl.actor --config_path config/train_so100_sim.json
```

For CPU, append `--policy.device=cpu --policy.storage_device=cpu` to **both**
commands and set `MUJOCO_GL=osmesa` on the actor. Paths in these examples are
relative to the repository root. The portable wrappers can also locate the
configuration from another directory:

```bash
bash scripts/train_sim.sh learner --policy.device=cpu
MUJOCO_GL=osmesa bash scripts/train_sim.sh actor --policy.device=cpu
```

The actor connects to `127.0.0.1:50051`, collects camera/state observations,
executes actions and sends transitions. The learner waits for at least 100
transitions before updating. Look for `Number of optimization step` in its
logs. `policy.online_steps` controls actor interactions; the learner continues
updating until interrupted. Stop both processes with Ctrl+C when finished.

By default, logs and checkpoints go under `outputs/train/<date>/<time>_SO100_sim_sac/`.
The learner writes `checkpoints/<step>/pretrained_model/` and a
`checkpoints/last` link at each save interval. The actor and learner resolve
separate timestamped log directories. If you set `--output_dir`, give each
process a separate directory: both validate that a new run's output does not
already exist. Avoid starting several learners on the same port.

The public config intentionally uses no offline dataset, no pretrained visual
encoder, no Hub upload, no W&B account, no torch compilation and **four
continuous action values** (XYZ + gripper). Images are scaled to [0, 1] by the
observation processor; the joint state remains in simulator units. The old
Franka normalization statistics do not apply to this SO100 observation.

This makes the pipeline self-contained. Sparse-reward learning from scratch can
be slow or unsuccessful without demonstrations and tuning; this configuration
is a software starting point rather than the original demo experiment recipe.

### Memory and run size

The public config stores up to 10,000 transitions in CPU RAM. Two 128 × 128
RGB streams require substantial memory. Lower the capacity and batch size on
smaller machines, and keep the warmup threshold at least as large as the batch:

```bash
# Apply the same overrides to learner and actor when relevant
--batch_size=8 --policy.online_buffer_capacity=1000 --policy.online_step_before_learning=32
```

Keep `policy.storage_device=cpu` to avoid filling GPU memory with replay images.
The required RAM/VRAM and useful training duration depend on configuration;
no minimum hardware requirement for converged training has been measured.

### Keyboard control and local demonstrations

The manual config is a **simulation** example, even though its entry point is
named `gym_manipulator`. It does not connect to hardware. On an X11 desktop:

```bash
python -m project.rl.gym_manipulator --config_path config/manual_so100.json

# Record your own demonstrations locally
python -m project.rl.gym_manipulator --config_path config/manual_so100.json \
  --mode=record --dataset.num_episodes_to_record=5
```

The dataset defaults to `local/so100-pick-cube` under `data/so100-pick-cube`.
Uploads are disabled. Use a fresh `--dataset.root` for each new collection.
Arrow keys move in XY, left/right Shift moves down/up, left/right Ctrl
closes/opens the gripper, Enter marks success, Esc ends a failed episode, and
R requests a re-record. Avoid pressing Esc solely to exit: Ctrl+C ends the
program. Replay a recorded episode with:

```bash
python -m project.rl.gym_manipulator --config_path config/manual_so100.json \
  --mode=replay --dataset.replay_episode=0
```

For optional human intervention while training, set
`--env.task=SO100PickCubeKeyboard-v0` on the actor in a graphical session.
The public keyboard config starts with human input enabled. For a custom
intervention policy, edit the wrapper's `require_intervention_key` setting;
Space toggles intervention when that mode is enabled. Manual collection,
keyboard input and real hardware require separate interactive verification.

To train with your own recorded data, copy the training config and replace
`dataset: null` with a compatible SO100 dataset configuration, for example:

```json
{
  "repo_id": "local/so100-pick-cube",
  "root": "data/so100-pick-cube",
  "video_backend": "pyav",
  "use_imagenet_stats": false
}
```

This is the **value of `dataset`**, not a complete training config. Keep camera
keys, six-state dimensions and four-action dimensions consistent with the
simulation. Validate recorded episodes before training; the original Franka
datasets are not SO100 demonstrations and must not be used as drop-in inputs.

## Troubleshooting

| Symptom | Check / action |
| :--- | :--- |
| `No module named project` | Activate the right environment and run the editable install; use `python -m pip`, not a different environment's pip. |
| `No module named gym_so100` | Reinstall this checkout; imports now use `project.rl.gym_so100` and require no custom `PYTHONPATH`. |
| `glGetError`, missing OSMesa or GL errors | Install rendering libraries and select `MUJOCO_GL=osmesa` or `egl` before Python starts. |
| `DISPLAY` / pynput error | Use an X11 desktop for keyboard control; use the default noninteractive task on a server. |
| Protobuf or gRPC version errors | Reinstall with the constraints. Generated code requires protobuf ≥6.31 and grpcio ≥1.73.1. |
| Model or dataset download during baseline run | Use the shipped config: `dataset=null`, `vision_encoder_name=null`. Optional backends may download assets. |
| Learner waits without optimization | Start actor with the same port/config; allow warmup to reach `online_step_before_learning`. Inspect both logs for a traceback. |
| Output directory already exists | Use a fresh directory per process; do not point actor and learner at the same new-run output directory. |
| CUDA unavailable or out of memory | Check `nvidia-smi`, install the matching CUDA torch wheel, reduce batch/capacity or use CPU. |
| TorchCodec warning | The baseline uses the PyAV fallback; installing TorchCodec is unnecessary for these checks. |
| Missing gamepad driver | Gamepad implementations/configuration are not included. Use the keyboard or headless task. |

### Other policies and real hardware

ACT, Diffusion, π₀, π₀.₅ and GR00T modules are research code retained from the
project's broader work. Their presence does not imply that the SAC actor and
learner train each of these models. A full VLA workflow needs its own model
weights, processor/tokenizer assets, compatible optional dependencies and
training/evaluation configuration. The `diffusion` extra installs diffusers;
it does not validate every VLA backend. The optional `tracking`, `realsense`
and `kinematics` extras are also outside the pinned SAC checks.

A real-robot experiment additionally needs the correct SO100 hardware revision,
serial ports, motor IDs and calibration, camera names/extrinsics, workspace
limits, any required URDF/kinematics assets, an action-space mapping from
simulation to hardware and a compatible trained policy. These artifacts are
not all supplied. Do not interpret the simulation JSON as a deployable
real-robot configuration. A complete real-robot result remains unverified.

For reproduction reports, include the Git commit, OS, rendering backend,
`python --version`, `python -m pip freeze`, config JSON, seed, and both actor
and learner logs. Remove account tokens or private paths before posting.
