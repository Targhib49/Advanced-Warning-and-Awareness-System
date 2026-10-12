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
    bit 3 = soft temperature limit; +16 for "has occurred since boot". Under-
    voltage and overheating both make a Pi behave strangely under load, and
    both are invisible unless you look for them. See describe_throttle().
    """
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


# vcgencmd get_throttled bits. The low half is "now", the high half (+16) is
# "has occurred since boot". Power and heat need different fixes, so name them.
_THROTTLE_BITS = {
    0: "under-voltage now", 1: "frequency capped now", 2: "throttled now",
    3: "soft temperature limit now",
    16: "under-voltage occurred", 17: "frequency capping occurred",
    18: "throttling occurred", 19: "soft temperature limit occurred",
}


def describe_throttle(flags: str) -> str:
    """Plain-language reading of a get_throttled value, with the likely fix."""
    try:
        value = int(flags, 16)
    except ValueError:
        return "unreadable"
    if value == 0:
        return "(0x0 = healthy)"
    names = [n for bit, n in _THROTTLE_BITS.items() if value & (1 << bit)]
    power = value & ((1 << 0) | (1 << 16))
    heat = value & ((1 << 3) | (1 << 19))
    hint = ("check the power supply" if power and not heat else
            "check the cooling" if heat and not power else
            "check the power supply and the cooling")
    return f"** {', '.join(names)}: {hint} **"


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
        lines.append(f"throttle {flags}  {describe_throttle(flags)}")
    return "\n".join(lines)
