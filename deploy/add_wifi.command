#!/bin/bash
# Add a Wi-Fi network to the AWAS Pi, as a backup it joins on its own.
#
# Double-click (or: bash deploy/add_wifi.command). Runs on the Mac that can
# already reach the Pi over SSH (`ssh awas`), while the Pi is on a network it
# knows. Whoever owns the new network types its name and password here; the
# password is never shown, never written to this Mac, and travels to the Pi
# over SSH's input rather than on a command line.
#
# The Pi keeps every network it is given and joins whichever is in range, so
# adding one does not move it off the network it is on now.
set -uo pipefail
PI="${AWAS_PI:-awas}"
IP="${AWAS_PI_IP:-}"          # when the .local name does not resolve (it can go
                              # stale on the Mac after switching networks)
pi() { ssh ${IP:+-o HostName=$IP} "$@"; }

echo "AWAS: add a Wi-Fi network to the Pi"
echo
if ! pi -n -o ConnectTimeout=8 -o BatchMode=yes "$PI" true 2>/dev/null; then
    echo "Cannot reach the Pi by name. If this Mac is on the same network as the Pi,"
    read -r -p "type the Pi's IP address (from the hotspot's device list), or Enter to stop: " IP
    if [ -z "$IP" ] || ! pi -n -o ConnectTimeout=8 -o BatchMode=yes "$PI" true 2>/dev/null; then
        echo "Cannot reach the Pi. Join the same network as the Pi first, then try again."
        read -r -p "Press Enter to close." _
        exit 1
    fi
fi

echo "Networks the Pi can see now:"
pi -n "$PI" 'nmcli -t -f SSID,SIGNAL dev wifi list --rescan yes 2>/dev/null | awk -F: '\''$1 != "" && !seen[$1]++ {printf "   %-32s signal %s%%\n", $1, $2}'\'' | head -15'
echo "   (a phone hotspot only shows up while its hotspot screen is open)"
echo
echo "Saved on the Pi already:"
pi -n "$PI" 'nmcli -t -f NAME,TYPE con show | awk -F: '\''$2 ~ /wireless/ {sub(/^netplan-wlan0-/, "", $1); print "   " $1}'\'''
echo

read -r -p "Network name (exactly, capitals and spaces matter): " SSID
if [ -z "$SSID" ]; then echo "No name given; nothing changed."; read -r -p "Press Enter to close." _; exit 1; fi
read -r -s -p "Password (hidden; leave empty for an open network): " PSK; echo
if [ -n "$PSK" ] && [ "${#PSK}" -lt 8 ]; then
    echo "A Wi-Fi password is at least 8 characters; nothing changed."
    read -r -p "Press Enter to close." _; exit 1
fi

# Name and password go in on stdin, one per line. Existing profiles are found
# by the SSID they connect to, not by their label: the Pi's first network is
# labelled "netplan-wlan0-<name>", and matching labels would add a duplicate.
printf '%s\n%s\n' "$SSID" "$PSK" | pi "$PI" '
read -r SSID; read -r PSK
active=$(nmcli -g GENERAL.CON-UUID dev show wlan0 2>/dev/null)
same=()
for uuid in $(nmcli -g UUID,TYPE con show | awk -F: '\''$2 == "802-11-wireless" {print $1}'\''); do
    [ "$(nmcli -g 802-11-wireless.ssid con show uuid "$uuid")" = "$SSID" ] || continue
    if [ "$uuid" = "$active" ]; then
        # Replacing it would drop the link this runs over, and a mistyped
        # password would leave the Pi on no network at all.
        exit 3
    fi
    same+=("$uuid")
done
for uuid in "${same[@]}"; do sudo nmcli connection delete uuid "$uuid" >/dev/null; done
if [ -n "$PSK" ]; then
    sudo nmcli connection add type wifi con-name "$SSID" ifname wlan0 ssid "$SSID" \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" connection.autoconnect yes >/dev/null
else
    sudo nmcli connection add type wifi con-name "$SSID" ifname wlan0 ssid "$SSID" \
        connection.autoconnect yes >/dev/null
fi
'
status=$?
unset PSK
echo
case $status in
    0) echo "Saved: the Pi will join \"$SSID\" by itself whenever it is in range."
       echo "To test it: turn the Pi's current network off and give it about a minute." ;;
    3) echo "\"$SSID\" is the network the Pi is on right now; it is already saved. Nothing changed." ;;
    *) echo "Could not save it on the Pi; nothing changed." ;;
esac
read -r -p "Press Enter to close." _
