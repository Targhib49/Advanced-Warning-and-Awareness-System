# AWAS simulator bridge

Connects Euro Truck Simulator 2 on a Windows laptop or a Mac to the AWAS
device (the Raspberry Pi). The Pi still does all the sensing and deciding; the laptop only
stands in for the truck and for the centre screen.

```
Windows laptop                                  Raspberry Pi
  ETS2 + scs-sdk-plugin ─┐                        driver camera  -> Layer 1
  awas_bridge.py ────────┼─ UDP 8097 ──────────▶  sim telemetry  -> speed, steering
     (telemetry + road view)                      road frames    -> lane, SDLP
  browser ◀──────────────┴─ http :8095 ────────── centre-screen app
```

## Set up a laptop (once per laptop, ~10 min plus the ETS2 download)

1. Install ETS2 from Steam and start it once.
2. Copy this `bridge` folder to the laptop and double-click **`install.bat`**.
   Run it as administrator only if ETS2 is under `C:\Program Files`.
   It installs Python if needed, the bridge's packages, and the telemetry
   plugin (scs-sdk-plugin V.1.12.1, checksum-pinned), and puts two shortcuts
   on the desktop.
3. Join the same phone hotspot as the Pi.

## Set up a Mac (once per Mac, ~5 min plus the ETS2 download)

ETS2 for Mac is an Intel build (it runs under Rosetta on Apple silicon), and
scs-sdk-plugin has no macOS version, so the Mac uses AWAS's own small plugin,
`mac/awas_telemetry.dylib`. It sends speed, steering, gear, signals, engine,
pause and speed limit to `127.0.0.1:8098` only; the bridge forwards them to
the Pi exactly as on Windows.

1. Install ETS2 from Steam.
2. Copy this `bridge` folder to the Mac and, in Terminal:

   ```
   bash bridge/mac/install_mac.sh            # or add  --pi <Pi's IP>
   ```

   It needs Python 3.9+ (python.org's installer, if the Mac has none) and
   writes only inside your account: a venv in
   `~/Library/Application Support/awas-bridge`, the plugin into
   `Euro Truck Simulator 2.app/Contents/MacOS/plugins/`, and two launchers on
   the Desktop. If a game update removes the plugin, run the installer again.
3. The first time the bridge runs, allow **Terminal** in System Settings >
   Privacy & Security > **Screen & System Audio Recording** (the road view),
   then quit Terminal and start the launcher again.

The plugin is prebuilt. After changing `mac/awas_telemetry.cpp`, rebuild it
with `mac/build_plugin.sh` (needs `xcode-select --install`). The SCS SDK
headers in `mac/scs_sdk` are SCS Software's, MIT-licensed
(`mac/scs_sdk/LICENSE.txt`).

## Each session

1. On the Pi, AWAS runs with `--sim` (and `--dashboard 8095`).
2. Start ETS2. Press **OK** on the "SDK features" dialog — that is the plugin.
   Run the game **windowed** (on the Mac, full screen also works), and set the camera to the bumper or hood view
   so the road fills the window without the cab.
3. Double-click **AWAS bridge (check)** (`.bat` on Windows, `.command` on
   the Mac). It prints speed, steering and,
   from the Pi, whether the lane is found:

   ```
   [  682 tel  410 frm  9 KB]   84.0 km/h  steer -0.02  engine on  |  Pi: receiving (407 frames)  lane -0.06 m, seen in 97% of frames
   ```
4. Open **http://awas.local:8095** in a browser beside the game (or the Pi's
   IP, if `.local` names don't resolve on this laptop). Keep the game window
   uncovered — it is the road camera.

## Controller

An Xbox controller works. In ETS2's controls, lower steering sensitivity and
add non-linearity. For the study, keep the same controller and settings for
every participant, and report it: a thumbstick steers more abruptly than a
wheel, which changes steering reversals and SDLP.

## Recording, for lane calibration

```
AWAS bridge.bat --record trip1 --duration 600
```

saves `road.avi` and `telemetry.jsonl`. Ten minutes of ordinary driving is
enough to calibrate the lane geometry for this camera view on the Pi
(`scripts/calibrate_lane.py`, saved as `config/lane_calibration_sim.json`).

## Without the game

`python awas_bridge.py --source mock --capture mock --check` sends a synthetic
drive and road, to test the link and the Pi end to end from any machine.
