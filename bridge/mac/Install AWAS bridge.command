#!/bin/bash
# Double-click to set up the AWAS bridge on this Mac (runs install_mac.sh).
# If macOS says it is from an unidentified developer: right-click > Open.
cd "$(dirname "$0")" && bash install_mac.sh "$@"
echo
read -r -p "Press Enter to close this window." _
