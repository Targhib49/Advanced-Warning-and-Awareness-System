#!/usr/bin/env bash
# Build "AWAS Demo.app" — a macOS wrapper so the demo can use the camera.
#
#   ./scripts/make_app.sh [args for run_webcam...]
#   open "AWAS Demo.app"
#
# WHY THIS EXISTS (macOS only — irrelevant on the Pi):
# macOS grants camera access per application bundle. A bare command-line
# process has no bundle and no NSCameraUsageDescription, so TCC refuses to even
# raise the prompt: AVCaptureDevice.requestAccess returns false immediately,
# records no decision, and the process never appears in System Settings. That
# is why no toggle can be found for it. Wrapping the same code in a real bundle
# that declares its purpose lets macOS ask the question normally.
#
# This is dev-environment scaffolding, not part of the product. Linux has no
# TCC; on the Pi the camera is just /dev/video0 and none of this applies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/AWAS Demo.app"
PY="$ROOT/venv/bin/python"
LOG="/tmp/awas-demo.log"

[ -x "$PY" ] || { echo "No venv at $PY — create it and install requirements first." >&2; exit 1; }

ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--driver default)

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>awas-demo</string>
  <key>CFBundleIdentifier</key><string>com.awas.demo</string>
  <key>CFBundleName</key><string>AWAS Demo</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSCameraUsageDescription</key><string>AWAS monitors driver alertness from the camera. Video is processed on this device and never leaves it.</string>
  <!-- The project may live under a TCC-protected folder (Documents, Desktop,
       Downloads). Without these the app cannot even read its own venv. -->
  <key>NSDocumentsFolderUsageDescription</key><string>AWAS reads its code, models and driver profiles from this folder.</string>
  <key>NSDesktopFolderUsageDescription</key><string>AWAS reads its code, models and driver profiles from this folder.</string>
  <key>NSDownloadsFolderUsageDescription</key><string>AWAS reads its code, models and driver profiles from this folder.</string>
  <key>NSMicrophoneUsageDescription</key><string>Reserved for the spoken-reply channel; unused in Phase 1.</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

{
  echo '#!/bin/bash'
  echo "cd \"$ROOT\" || exit 1"
  printf 'exec "%s" -m scripts.run_webcam' "$PY"
  for a in "${ARGS[@]}"; do printf ' %q' "$a"; done
  printf ' >>"%s" 2>&1\n' "$LOG"
} > "$APP/Contents/MacOS/awas-demo"

chmod +x "$APP/Contents/MacOS/awas-demo"
codesign --force --sign - "$APP" >/dev/null 2>&1

echo "built: $APP"
echo "  args: ${ARGS[*]}"
echo "  log:  $LOG"
echo
echo "run it with:   open \"$APP\""
echo "watch output:  tail -f $LOG"
echo
echo "FIRST RUN ONLY, if this project lives under Documents/Desktop/Downloads:"
echo "  macOS will not raise a folder-access prompt for a script-based bundle,"
echo "  so the app cannot read its own code until you add it manually to"
echo "  System Settings > Privacy & Security > Full Disk Access (+ button, or"
echo "  drag the .app in). Quit and relaunch the app afterwards."
echo "  Moving the project outside those folders avoids this entirely."
