# Baseline dependencies

`pyproject.toml` pins the direct simulation dependencies. `constraints-py310.txt`
additionally pins their resolved Python dependencies, plus the `keyboard` and
`dev` extras, for **Python 3.10 on Linux x86_64**. A constraints file limits
versions; it does not install every listed package. Install it with:

```bash
python -m pip install -c requirements/constraints-py310.txt -e ".[keyboard,dev]"
```

Install torch 2.7.1 and torchvision 0.22.1 first from the CPU or CUDA 12.6 index
in the root README. They are intentionally absent from the constraints so that
`2.7.1+cpu` and `2.7.1+cu126` can both satisfy the project requirement. NVIDIA
runtime dependencies are specific to the CUDA wheel and are outside this CPU
lock. OS libraries, the driver and FFmpeg are documented separately.

To refresh the constraints deliberately, use uv (the initial resolution used
uv 0.13.7) in a Python 3.10 environment:

```bash
uv pip compile pyproject.toml --extra dev --extra keyboard --torch-backend cpu \
  --index-url https://pypi.org/simple --output-file /tmp/so100-resolved.txt
python - <<'PY'
from pathlib import Path
lines = Path('/tmp/so100-resolved.txt').read_text().splitlines()
pins = [line for line in lines if line and not line.startswith(
    (' ', '#', '--', 'torch==', 'torchvision=='))]
Path('requirements/constraints-py310.txt').write_text(
    '# Python 3.10, Linux x86_64; CPU simulation + keyboard + dev dependencies.\n'
    '# Generated with uv pip compile; see requirements/README.md.\n'
    '# PyTorch is installed separately to choose CPU or CUDA wheels.\n'
    + '\n'.join(pins) + '\n')
PY
```

Run `pip check`, the offline smoke check, tests, and wheel installation checks
before accepting an updated resolution. Optional tracking, RealSense,
kinematics and Diffusion extras are not part of this locked baseline.
