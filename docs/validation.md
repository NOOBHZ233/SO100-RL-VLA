# Local validation record

Validated on **2026-11-11 (Asia/Shanghai)** against this working tree, based on
commit `d3a8cf3`. This is a record of executed checks, not a claim that a remote
GitHub Actions run or a real-robot experiment has completed.

| Check | Result |
| :--- | :--- |
| Fresh Python environment | Python 3.10.19 venv created without system site packages; CPU torch 2.7.1+cpu and torchvision 0.22.1+cpu installed with the pinned Python stack |
| Documented pip install | Editable install using `requirements/constraints-py310.txt` and the `keyboard,dev` extras succeeded; `pip check` reported no broken requirements |
| Conda specification | `conda env create --dry-run -f environment.yml` resolved Python 3.10.19, pip 25.3 and FFmpeg 7.1.1; the complete Conda environment was not installed locally |
| CPU rendering | Both EGL and OSMesa smoke checks passed on Ubuntu 20.04 x86_64; OSMesa used an extracted Ubuntu `libosmesa6` package for local verification |
| CUDA execution | Smoke check passed on RTX 3060 Laptop GPU (6 GiB), driver 570.133.20, torch 2.7.1+cu126 / torchvision 0.22.1 in the existing GPU environment; this was not a fresh CUDA installation |
| Integration tests | **10 passed**: offline SAC inference/updates, checkpoint reload, seeded Gymnasium behavior, camera dimensions, config parsing, optional imports, intervention actions, recording/replay, gripper penalty and actor–learner CLI transport |
| Separate actor/learner | Actual CPU CLI processes exchanged parameters and transitions over gRPC, performed optimization, saved checkpoints and exported replay data; short partial episodes were also transmitted and the actor exited normally |
| Local recording | A headless 100-frame episode was recorded with two camera videos and replayed through the CLI; the automated regression also reloads and decodes a short recorded dataset |
| Distributions | Wheel and sdist built; wheel contains both MJCF XML files and all 18 STL assets, with no bytecode; installed-wheel smoke check runs outside the checkout |
| Entry points | Actor, learner and gym manipulator `--help` commands load without optional VLA model dependencies |
| Documentation | English and Chinese README layouts inspected in a local GitHub-style preview; local links, SVG syntax and shell script syntax checked |

Warnings observed in tests: unbounded joint observation boxes in Gymnasium,
NumPy conversion warnings during dataset statistics/scalar conversion, and
upstream torchvision video API deprecation. The baseline pins the tested
versions. PyAV is used when TorchCodec is absent.

The GitHub Actions template is supplied in `docs/reproducibility-workflow.yml`;
it must be copied to `.github/workflows/reproducibility.yml` to enable CI.
No remote CI result is asserted here. The interactive X11 keyboard listener, robot motors/cameras,
long-run convergence, demo success rates, and ACT/Diffusion/π/GR00T training
have not been validated. The original trained policy and demonstration data
are still needed for reproduction of the published real-robot footage.
