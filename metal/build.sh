#!/bin/bash
# Build the Metal PoC (host + runtime-compiled MSL).
set -euo pipefail
cd "$(dirname "$0")"
clang++ -std=c++20 -O2 -fno-objc-arc \
  metal_bench.mm \
  -I ../qiner-mac/src \
  -framework Metal -framework Foundation \
  -o metal_bench
echo "built: ./metal_bench (validate | bench [N])"
