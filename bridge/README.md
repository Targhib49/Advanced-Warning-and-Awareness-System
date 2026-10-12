# AWAS simulator bridge

Connects Euro Truck Simulator 2 on a Windows laptop to the AWAS device (the
Raspberry Pi). The Pi still does all the sensing and deciding; the laptop only
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

## Each session

1. On the Pi, AWAS runs with `--sim` (and `--dashboard 8095`).
2. Start ETS2. Press **OK** on the "SDK features" dialog — that is the plugin.
   Run the game **windowed**, and set the camera to the bumper or hood view
   so the road fills the window without the cab.
3. Double-click **AWAS bridge (check)**. It prints speed, steering and,
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
