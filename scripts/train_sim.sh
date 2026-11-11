#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
role="${1:-}"
if [[ "$role" != learner && "$role" != actor ]]; then
  echo "Usage: bash scripts/train_sim.sh {learner|actor} [config overrides...]" >&2
  exit 2
fi
shift
cd "$repo_root"
exec python -m "project.rl.$role" --config_path "$repo_root/config/train_so100_sim.json" "$@"
