#!/bin/bash
# Run the device profiler under Terminal.app, which holds the camera grant.
# Results land in artifacts/host_profile.json so they can be read back.
#
# WHY: macOS attributes camera permission to the launching app. Embedded
# terminals inside AI coding apps never get that attribution. Irrelevant on the
# Pi — Linux has no TCC.
cd "$(dirname "$0")/.." || exit 1
mkdir -p artifacts
echo "AWAS device profiler — sit facing the camera."
echo
exec ./venv/bin/python -m scripts.profile_device \
    --camera 0 --frames 200 --audio-repeats 8 --countdown 6 \
    --out artifacts/host_profile.json 2>&1 | tee artifacts/host_profile.log
