#!/bin/bash
# AWAS simulator bridge: one-shot setup for a Mac with ETS2 (the macOS twin of
# install_windows.ps1). From a Terminal, in this folder:
#
#     bash install_mac.sh                  # the Pi is found as awas.local
#     bash install_mac.sh --pi 10.x.x.x    # if .local names don't resolve
#
# What it does, all idempotent and all inside your user account (no sudo):
#   1. A private venv in ~/Library/Application Support/awas-bridge
#   2. The AWAS telemetry plugin into ETS2's plugins folder
#   3. "AWAS bridge" and "AWAS bridge (check)" launchers on the Desktop
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BRIDGE="$(dirname "$HERE")"
HOME_="$HOME/Library/Application Support/awas-bridge"
PI=""
APP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pi)  PI="$2"; shift 2 ;;
        --app) APP="$2"; shift 2 ;;      # the .app, if Steam keeps it somewhere unusual
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

say()  { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
warn() { printf '    \033[33m%s\033[0m\n' "$1"; }

# ------------------------------------------------------------------ python
say "Python"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    exe="$(command -v "$c" 2>/dev/null)" || continue
    # Apple's /usr/bin/python3 is a stub that pops up an installer dialog
    # when the Command Line Tools are missing; skip it rather than block.
    if [ "$exe" = /usr/bin/python3 ] && ! xcode-select -p >/dev/null 2>&1; then continue; fi
    if "$exe" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then PY="$exe"; break; fi
done
if [ -z "$PY" ]; then
    warn "No Python 3.9+ found. Install Python 3.12 from https://www.python.org/downloads/macos/"
    warn "then run this again."
    exit 1
fi
echo "    using: $("$PY" -V) ($PY)"

# ------------------------------------------------------------------ venv
say "Bridge environment in $HOME_"
mkdir -p "$HOME_"
[ -x "$HOME_/venv/bin/python" ] || "$PY" -m venv "$HOME_/venv"
"$HOME_/venv/bin/python" -m pip install -q --upgrade pip
"$HOME_/venv/bin/python" -m pip install -q -r "$BRIDGE/requirements.txt"
cp "$BRIDGE/awas_bridge.py" "$HOME_/awas_bridge.py"
echo "    packages installed"

# ------------------------------------------------------------------ ETS2
say "Euro Truck Simulator 2"
find_ets2() {
    local steam="$HOME/Library/Application Support/Steam"
    local libs=("$steam")
    local vdf="$steam/steamapps/libraryfolders.vdf"
    if [ -f "$vdf" ]; then
        while IFS= read -r p; do libs+=("$p"); done < <(sed -n 's/^[[:space:]]*"path"[[:space:]]*"\(.*\)"/\1/p' "$vdf")
    fi
    for lib in "${libs[@]}"; do
        local app="$lib/steamapps/common/Euro Truck Simulator 2/Euro Truck Simulator 2.app"
        if [ -x "$app/Contents/MacOS/eurotrucks2" ]; then echo "$app"; return 0; fi
    done
    return 1
}
[ -n "$APP" ] || APP="$(find_ets2)" || true
if [ -z "$APP" ] || [ ! -d "$APP/Contents/MacOS" ]; then
    warn "ETS2 not found through Steam. Install it, then run this again, or pass"
    warn "--app \"/path/to/Euro Truck Simulator 2.app\"."
else
    echo "    found: $APP"
    # The game loads every library in "plugins" next to its binary.
    PLUGINS="$APP/Contents/MacOS/plugins"
    mkdir -p "$PLUGINS"
    cp "$HERE/awas_telemetry.dylib" "$PLUGINS/awas_telemetry.dylib"
    # A folder downloaded through a browser is quarantined; the game would
    # then refuse to load the plugin without saying why.
    xattr -d com.apple.quarantine "$PLUGINS/awas_telemetry.dylib" 2>/dev/null || true
    echo "    telemetry plugin installed: $PLUGINS/awas_telemetry.dylib"
fi

# ------------------------------------------------------------------ launchers
say "Desktop launchers"
[ -n "$PI" ] && echo "$PI" > "$HOME_/pi" || true
cat > "$HOME_/run.sh" <<EOF
#!/bin/bash
# Started by the Desktop launchers. The Pi's address, if not awas.local, is
# the one line in: $HOME_/pi
PI_ARGS=()
[ -s "$HOME_/pi" ] && PI_ARGS=(--pi "\$(head -1 "$HOME_/pi")")
"$HOME_/venv/bin/python" "$HOME_/awas_bridge.py" "\${PI_ARGS[@]}" "\$@"
EOF
chmod +x "$HOME_/run.sh"
printf '#!/bin/bash\n"%s/run.sh"\n'         "$HOME_" > "$HOME/Desktop/AWAS bridge.command"
printf '#!/bin/bash\n"%s/run.sh" --check\n' "$HOME_" > "$HOME/Desktop/AWAS bridge (check).command"
chmod +x "$HOME/Desktop/AWAS bridge.command" "$HOME/Desktop/AWAS bridge (check).command"
echo "    'AWAS bridge' and 'AWAS bridge (check)' are on the Desktop"

say "Done"
cat <<EOF
    1. Start ETS2. It shows an 'SDK features' dialog the first time: press OK.
    2. Join the phone hotspot the Pi is on.
    3. Double-click 'AWAS bridge (check)' to see the speed, steering and lane.
       The first time, macOS asks to let Terminal record the screen (that is
       the road view): allow it in System Settings > Privacy & Security >
       Screen & System Audio Recording, then quit Terminal and start again.
       If 'awas.local' is not found:  bash install_mac.sh --pi <Pi's IP>
EOF
