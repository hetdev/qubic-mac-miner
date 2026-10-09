#!/bin/bash
# Build the Metal PoC + GPU solo runner (runtime-compiled MSL).
set -euo pipefail
cd "$(dirname "$0")"

CXXFLAGS="-std=c++20 -O2 -fno-objc-arc"
INC="-I ../qiner-mac/src"
FRAMEWORKS="-framework Metal -framework Foundation"

clang++ $CXXFLAGS $INC metal_bench.mm $FRAMEWORKS -o metal_bench
echo "built: ./metal_bench  (validate | bench [N])"

clang++ $CXXFLAGS $INC gpu_miner.mm ../qiner-mac/src/keyUtils.cpp $FRAMEWORKS -o gpu_miner
echo "built: ./gpu_miner    (GPU solo miner — see README)"
