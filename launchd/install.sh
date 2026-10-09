#!/bin/bash
# LaunchAgent installer (equivalent to: scripts/qubicminer install-service)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime"
"$ROOT/scripts/qubicminer" install-service
