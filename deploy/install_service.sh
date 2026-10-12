#!/usr/bin/env bash
# Install AWAS as a boot service, and the dashboard as the desktop's kiosk.
#
#     ./deploy/install_service.sh            # install (or update) and start
#     ./deploy/install_service.sh --no-kiosk # service only, no desktop browser
#     ./deploy/install_service.sh --remove   # stop and uninstall both
#
# Run as the user AWAS should run as; it uses sudo where it must. Idempotent,
# and it never overwrites an edited /etc/default/awas.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
USER_ID="$(id -u)"
UNIT=/etc/systemd/system/awas.service
DEFAULTS=/etc/default/awas
# An XDG autostart entry, not a line in ~/.config/labwc/autostart: a user
# labwc autostart file replaces the system one, and with it the desktop and
# panel. XDG entries are run by the system autostart in addition.
AUTOSTART="$HOME/.config/autostart/awas-kiosk.desktop"

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

if [ "${1:-}" = "--remove" ]; then
    say "removing"
    sudo systemctl disable --now awas 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    rm -f "$AUTOSTART"
    echo "    removed (kept $DEFAULTS)"
    exit 0
fi

[ -x "$ROOT/venv/bin/python" ] || { echo "no venv at $ROOT/venv — run scripts/setup_pi.sh first"; exit 1; }

say "service"
sed -e "s|@USER@|$USER_NAME|g" -e "s|@UID@|$USER_ID|g" -e "s|@ROOT@|$ROOT|g" \
    "$ROOT/deploy/awas.service" | sudo tee "$UNIT" >/dev/null
if [ ! -f "$DEFAULTS" ]; then
    sudo install -m 644 "$ROOT/deploy/awas.default" "$DEFAULTS"
    echo "    settings: $DEFAULTS (new)"
else
    echo "    settings: $DEFAULTS (kept as edited)"
fi
grep -E '^AWAS_ARGS=' "$DEFAULTS" | sed 's/^/    /'

# Audio over PipeWire needs the user's services running without a login.
sudo loginctl enable-linger "$USER_NAME"

sudo systemctl daemon-reload
sudo systemctl enable awas >/dev/null
sudo systemctl restart awas
sleep 3
echo "    awas: $(systemctl is-active awas)"

if [ "${1:-}" != "--no-kiosk" ]; then
    say "dashboard kiosk"
    chmod +x "$ROOT/deploy/kiosk.sh"
    mkdir -p "$(dirname "$AUTOSTART")"
    cat > "$AUTOSTART" <<EOF
[Desktop Entry]
Type=Application
Name=AWAS dashboard
Comment=The AWAS centre-screen app, full screen
Exec=$ROOT/deploy/kiosk.sh
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
    echo "    $AUTOSTART (starts with the desktop)"
fi

say "done"
cat <<EOF
    Starts on every boot. Useful commands:
      journalctl -u awas -f              watch it
      sudo systemctl restart awas        after editing $DEFAULTS
      sudo systemctl stop awas           e.g. to use the camera for something else
      ./deploy/install_service.sh --remove
EOF
