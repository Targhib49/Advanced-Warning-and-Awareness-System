#!/bin/bash
# Double-click this, or `open` it, to run the demo under Terminal.app.
#
# WHY: macOS attributes camera permission to the app that launched the process.
# Embedded terminals inside AI coding apps do not get that attribution — the
# request is refused without ever raising a prompt, and nothing appears in
# System Settings to toggle. Terminal.app does get it. Same code, right host.
# (None of this applies on the Pi; Linux has no TCC.)
cd "$(dirname "$0")/.." || exit 1

# Defaults, only when no arguments were passed. Setting the positional
# parameters keeps them as separate words — a quoted "${@:-a b}" default
# collapses into ONE argument containing a space.
if [ $# -eq 0 ]; then
    set -- --driver targhib
fi

echo "AWAS demo — accept the camera prompt when it appears."
echo "running: python -m scripts.run_webcam $*"
echo
exec ./venv/bin/python -m scripts.run_webcam "$@"
