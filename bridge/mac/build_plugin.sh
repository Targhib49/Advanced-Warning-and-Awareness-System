#!/bin/bash
# Build the AWAS ETS2 telemetry plugin for macOS. Needs the Xcode Command Line
# Tools (xcode-select --install). The prebuilt awas_telemetry.dylib next to
# this script is what install_mac.sh uses; rebuild only after changing the .cpp.
set -euo pipefail
cd "$(dirname "$0")"

# Apple's clang, not a Homebrew one: it is the toolchain that knows the SDK.
/usr/bin/clang++ -std=c++17 -O2 -Wall -Wno-unused-function \
    -arch x86_64 -mmacosx-version-min=10.15 \
    -dynamiclib \
    -Wl,-exported_symbol,_scs_telemetry_init -Wl,-exported_symbol,_scs_telemetry_shutdown \
    -I scs_sdk/include \
    -o awas_telemetry.dylib awas_telemetry.cpp
codesign --force --sign - awas_telemetry.dylib      # ad-hoc; the game allows unsigned plugins
file awas_telemetry.dylib
nm -gU awas_telemetry.dylib
