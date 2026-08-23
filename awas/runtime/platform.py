"""Platform facts the runtime needs to behave differently on a Pi (plan §7.2).

Kept in one place so the rest of the code never branches on `sys.platform`
directly. Everything here degrades to "unknown" rather than raising: a missing
thermal sensor is a reason to skip thermal management, not to refuse to drive.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Where a Linux SoC usually exposes its die temperature, in millidegrees.
_THERMAL_ZONES = (
    Path("/sys/class/thermal/thermal_zone0/temp"),
    Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
)


@dataclass(frozen=True)
class DeviceInfo:
    system: str
    machine: str
    python: str
    is_pi: bool
    model: str = ""
    cores: int = 0

    @property
    def label(self) -> str:
        return self.model or f"{self.system} {self.machine}"


def device_info() -> DeviceInfo:
    model = ""
    pi = False
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            model = dt_model.read_text(errors="ignore").strip("\x00 \n")
            pi = "raspberry pi" in model.lower()
        except OSError:
            pass
    return DeviceInfo(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        is_pi=pi,
        model=model,
        cores=len(__import__("os").sched_getaffinity(0))
        if hasattr(__import__("os"), "sched_getaffinity")
        else (__import__("os").cpu_count() or 0),
    )


def soc_temperature_c() -> Optional[float]:
    """Die temperature in Celsius, or None where it cannot be read.

    Used by the degradation ladder: past the thermal ceiling the runtime drops
    frame rate rather than letting the kernel throttle it unpredictably. A
    silently throttled safety system is a broken one (plan §5.4).
    """
    for zone in _THERMAL_ZONES:
        try:
            raw = int(zone.read_text().strip())
            return raw / 1000.0 if raw > 1000 else float(raw)
        except (OSError, ValueError):
            continue

    if sys.platform == "darwin":
        return None      # no unprivileged interface; not worth shelling out
    return None


def throttled_flags() -> Optional[str]:
    """Raspberry Pi throttling state via vcgencmd, when available.

    Bit 0 = under-voltage now, bit 1 = frequency capped, bit 2 = throttled,
    bit 16 = under-voltage has occurred since boot. Under-voltage is the most
    common cause of a Pi behaving strangely under load, and it is invisible
    unless you look for it.
    """
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def describe() -> str:
    d = device_info()
    lines = [
        f"device   {d.label}",
        f"system   {d.system} {d.machine}, python {d.python}, {d.cores} cores",
    ]
    t = soc_temperature_c()
    lines.append(f"temp     {t:.1f} C" if t is not None else "temp     unavailable")
    flags = throttled_flags()
    if flags:
        lines.append(f"throttle {flags}"
                     + ("  (0x0 = healthy)" if flags in ("0x0", "0") else
                        "  ** non-zero: check the power supply **"))
    return "\n".join(lines)
