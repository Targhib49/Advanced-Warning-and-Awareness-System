#!/usr/bin/env bash
# Fetch the face landmark model. mediapipe ships none from 0.10 onward.
#
#   ./scripts/fetch_models.sh
#
# Re-run on the Pi as part of setup; the checksum is pinned so a silently
# changed upstream model does not become a silently changed detector.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$DIR"

# name | url | sha256 | approx size, for the progress message
MODELS=(
  "face_landmarker.task|https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task|64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff|3.6 MB"
  "efficientdet_lite0.tflite|https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite|40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58|13.2 MB"
)

sha_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d" " -f1
  else sha256sum "$1" | cut -d" " -f1; fi
}

for entry in "${MODELS[@]}"; do
  IFS="|" read -r NAME URL SHA SIZE <<< "$entry"
  FILE="$DIR/$NAME"

  if [ ! -f "$FILE" ]; then
    echo "fetching $NAME (~$SIZE)"
    curl -sSL -o "$FILE" "$URL"
  fi

  GOT="$(sha_of "$FILE")"
  if [ "$GOT" != "$SHA" ]; then
    echo "CHECKSUM MISMATCH for $NAME" >&2
    echo "  expected $SHA" >&2
    echo "  got      $GOT" >&2
    echo "Refusing to vouch for this file. Delete it and re-run, or update the" >&2
    echo "hash deliberately if you intended to move to a new model version." >&2
    exit 1
  fi
  echo "ok: $FILE"
done
