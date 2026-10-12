#!/bin/sh
# The centre-screen app, full screen on the Pi's own desktop. Started by the
# desktop session (install_service.sh adds it to labwc's autostart).
#
# Waits for the AWAS service to answer first: a browser that loads before the
# server shows an error page and never retries.

URL="http://localhost:8095/"
for _ in $(seq 1 60); do
    curl -fs -o /dev/null "$URL" && break
    sleep 2
done

# A dedicated profile, so no session-restore, sign-in or translate prompt ever
# covers the display.
exec chromium --kiosk --start-fullscreen --no-first-run --noerrdialogs --disable-infobars \
    --disable-features=Translate --disable-session-crashed-bubble --password-store=basic \
    --user-data-dir="$HOME/.config/awas-kiosk" --ozone-platform=wayland "$URL"
