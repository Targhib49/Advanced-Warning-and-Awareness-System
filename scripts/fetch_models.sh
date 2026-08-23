#!/usr/bin/env bash
# Fetch the face landmark model. mediapipe ships none from 0.10 onward.
#
#   ./scripts/fetch_models.sh
#
# Re-run on the Pi as part of setup; the checksum is pinned so a silently
# changed upstream model does not become a silently changed detector.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
FILE="$DIR/face_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

mkdir -p "$DIR"

if [ -f "$FILE" ]; then
  echo "already present: $FILE"
else
  echo "fetching face_landmarker.task (~3.6 MB)"
  curl -sSL -o "$FILE" "$URL"
fi

if command -v shasum >/dev/null 2>&1; then
  GOT="$(shasum -a 256 "$FILE" | cut -d' ' -f1)"
else
  GOT="$(sha256sum "$FILE" | cut -d' ' -f1)"
fi

if [ "$GOT" != "$SHA" ]; then
  echo "CHECKSUM MISMATCH" >&2
  echo "  expected $SHA" >&2
  echo "  got      $GOT" >&2
  echo "Refusing to vouch for this file. Delete it and re-run, or update SHA" >&2
  echo "deliberately if you intended to move to a new model version." >&2
  exit 1
fi

echo "ok: $FILE"
