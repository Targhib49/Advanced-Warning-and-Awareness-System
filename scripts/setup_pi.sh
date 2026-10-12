#!/usr/bin/env bash
# One-shot provisioning for a Raspberry Pi 5 running Pi OS Bookworm (64-bit).
#
#     git clone <repo> ~/awas && cd ~/awas
#     ./scripts/setup_pi.sh
#
# Idempotent: safe to re-run. Everything it installs is offline-capable once
# installed — the Pi needs a network for setup and never again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }
warn() { printf "\033[33m    %s\033[0m\n" "$1"; }

# ---------------------------------------------------------------- sanity
say "checking the machine"
ARCH="$(uname -m)"
echo "    arch: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
  warn "expected aarch64 (64-bit Pi OS). A 32-bit install will not find"
  warn "mediapipe wheels at all. Reflash with the 64-bit image if so."
fi
[ -r /proc/device-tree/model ] && echo "    model: $(tr -d '\0' < /proc/device-tree/model)"
echo "    python: $(python3 -V)"

# ---------------------------------------------------------------- system
say "system packages"
sudo apt-get update -qq
# libportaudio2  -> sounddevice
# libgl1, libglib -> opencv
# No BLAS package: numpy's aarch64 wheels bundle OpenBLAS, and
# libatlas-base-dev no longer exists from Trixie on.
# Trixie renamed libglib2.0-0 to libglib2.0-0t64 (the 64-bit time_t transition).
GLIB=libglib2.0-0
apt-cache show libglib2.0-0t64 >/dev/null 2>&1 && GLIB=libglib2.0-0t64
sudo apt-get install -y -qq \
  python3-venv python3-dev \
  libportaudio2 \
  libgl1 "$GLIB" \
  git curl

# ---------------------------------------------------------------- python
say "python environment"
# mediapipe 0.10.18 publishes aarch64 wheels for CPython 3.9-3.12 only, and
# Trixie ships 3.13. On a newer system Python, build the venv on a 3.12 that
# uv fetches instead, leaving the system interpreter alone.
py_ok() { "$1" -c 'import sys; sys.exit(not (3, 9) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; }
if [ -d venv ] && ! py_ok ./venv/bin/python; then
  warn "existing venv is on $(./venv/bin/python -V 2>&1), which mediapipe 0.10.18 cannot use; rebuilding"
  rm -rf venv
fi
if [ ! -d venv ]; then
  if py_ok python3; then
    python3 -m venv venv
  else
    echo "    system $(python3 -V) is too new for mediapipe 0.10.18; using Python 3.12 from uv"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
      sudo apt-get install -y -qq pipx
      pipx install uv
    fi
    uv python install 3.12
    uv venv --python 3.12 --seed venv
  fi
fi
echo "    venv python: $(./venv/bin/python -V)"
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements-pi.txt
echo "    mediapipe: $(./venv/bin/python -c 'import mediapipe as m; print(m.__version__)' 2>/dev/null || echo FAILED)"

# ---------------------------------------------------------------- assets
say "landmark model"
./scripts/fetch_models.sh

say "phrase bank"
if command -v piper >/dev/null 2>&1; then
  if [ -n "${PIPER_VOICE:-}" ]; then
    ./venv/bin/python -m scripts.render_phrases --piper-model "$PIPER_VOICE"
  else
    warn "piper is installed but PIPER_VOICE is not set. Download a voice, then:"
    warn "  PIPER_VOICE=/path/voice.onnx ./scripts/setup_pi.sh"
  fi
else
  warn "piper not installed — the system will run silent until it is."
  warn "  Install piper, download a voice, then re-run this script."
  warn "  Alternatively, copy assets/phrases/ from a machine that has rendered it;"
  warn "  the WAVs are portable and that is a perfectly good shortcut."
fi

# ---------------------------------------------------------------- checks
say "camera"
if [ -e /dev/video0 ]; then
  echo "    /dev/video0 present"
else
  warn "no /dev/video0. For a USB camera check it is plugged in; for the CSI"
  warn "module check 'libcamera-hello' works and that the ribbon is seated."
fi

say "verification"
./venv/bin/python -m pytest tests/ -q || warn "tests failed — do not deploy this"

say "device profile"
./venv/bin/python -m scripts.profile_device --frames 60 --out artifacts/pi_profile.json || true

cat <<'DONE'

==> next

  1. VERIFY THE EYE LANDMARK INDICES on this mediapipe version:
         ./venv/bin/python -m scripts.profile_device --landmarks --camera 0
     The EAR median must land between about 0.10 and 0.60. Anything else means
     the index sets moved between releases and every number downstream is
     meaningless. This is the single most important check on a new machine.

  2. Run it:
         ./venv/bin/python -m scripts.run_webcam --driver <name>

  3. Install as a service, if it should start on boot:
         sudo cp deploy/awas.service /etc/systemd/system/
         sudo systemctl daemon-reload && sudo systemctl enable --now awas

DONE
