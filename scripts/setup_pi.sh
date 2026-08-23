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
# libatlas-base-dev -> numpy BLAS
sudo apt-get install -y -qq \
  python3-venv python3-dev \
  libportaudio2 libatlas-base-dev \
  libgl1 libglib2.0-0 \
  git curl

# ---------------------------------------------------------------- python
say "python environment"
if [ ! -d venv ]; then
  python3 -m venv venv
fi
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
