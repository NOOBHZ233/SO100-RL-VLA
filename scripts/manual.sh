#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python -m project.rl.gym_manipulator --config_path "$repo_root/config/manual_so100.json" "$@"
