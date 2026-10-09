#!/bin/bash
# Native macOS build (arm64 and x86_64) without cmake.
# Usage: ./build-macos.sh [arm64|x86_64|universal]
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${1:-arm64}"
OUT=build-macos
mkdir -p "$OUT"

CXXFLAGS="-std=c++20 -O3 -DNDEBUG"

build_target() {
  local target="$1"; shift
  local suffix="$1"; shift
  echo "== Qiner ($suffix) =="
  clang++ $CXXFLAGS "$@" src/Qiner.cpp src/keyUtils.cpp -pthread -o "$OUT/Qiner$suffix"
  echo "== AntMiner ($suffix) =="
  clang++ $CXXFLAGS "$@" -Isrc -Isrc/ant src/ant/AntMiner.cpp src/keyUtils.cpp -pthread -o "$OUT/AntMiner$suffix"
  echo "== qiner_test ($suffix) =="
  clang++ $CXXFLAGS "$@" -Isrc -Iexternal/catch2/include \
    -DQINER_TEST_DATA_DIR='"test/data"' \
    test/main.cpp test/bpp9000_ant_colony_test.cpp -pthread -o "$OUT/qiner_test$suffix"
}

case "$ARCH" in
  arm64)    build_target arm64 ""      -Isrc -arch arm64 ;;
  x86_64)   build_target x86_64 "_x64" -Isrc -arch x86_64 ;;
  universal)
    build_target universal "" -Isrc -arch arm64 -arch x86_64
    # fat binary for distribution
    lipo -create "$OUT/Qiner" -output /dev/null 2>/dev/null || true
    ;;
  *) echo "unsupported architecture: $ARCH (use arm64 | x86_64 | universal)"; exit 1 ;;
esac

echo
echo "Binaries in $OUT/:"
ls -la "$OUT" | grep -E 'Qiner|AntMiner|qiner_test' || true
echo
echo "Validation suite:  ./$OUT/qiner_test"
