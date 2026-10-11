#!/usr/bin/env bash
# Lint and test, as CI would. Run from the repository root or inside the container.
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check .
pytest -q
