"""Who drives this vehicle: the registry behind "Who's driving?" (plan §8).

Each driver has an id (a slug of their name, used in file names), a display
name, when they consented, and the result of their last guided calibration.
Their eye profile lives beside it in `profiles/<id>.json`; their session logs
are `sessions/*_<id>_*.jsonl`.

Deleting a driver removes all three. §8 is explicit that biometric-derived
data stays on the device and the driver can delete it; a delete that left the
session logs behind would not be one.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

NAME_MAX = 24
_NAME_OK = re.compile(r"^[\w .'\-]+$", re.UNICODE)
# Ids that would collide with what the system itself uses.
_RESERVED = {"default", "bench", "setup", "drivers"}


class DriverNameError(ValueError):
    pass


@dataclass
class Driver:
    id: str
    name: str
    created_at: float = field(default_factory=time.time)
    consent_at: Optional[float] = None
    last_used_at: Optional[float] = None
    # From the last guided calibration (awas.driver.guided); None until one passes.
    eye_signal: Optional[str] = None
    calibration: Optional[dict] = None

    @property
    def calibrated(self) -> bool:
        return self.eye_signal is not None


def clean_name(raw: str) -> str:
    name = " ".join(str(raw).split())
    if not name:
        raise DriverNameError("Enter a name.")
    if len(name) > NAME_MAX:
        raise DriverNameError(f"Use at most {NAME_MAX} characters.")
    if not _NAME_OK.match(name):
        raise DriverNameError("Use letters, numbers, spaces, . ' and - only.")
    return name


def slugify(name: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")
    return slug or "driver"


class DriverRegistry:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "drivers.json"
        self.profiles_dir = self.data_dir / "profiles"
        self.sessions_dir = self.data_dir / "sessions"
        self._drivers: Dict[str, Driver] = self._load()

    # ---------------- persistence ----------------

    def _load(self) -> Dict[str, Driver]:
        if not self.path.exists():
            return {}
        blob = json.loads(self.path.read_text())
        return {d["id"]: Driver(**d) for d in blob.get("drivers", [])}

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"drivers": [asdict(d) for d in self._drivers.values()]}, indent=2))
        tmp.replace(self.path)          # never a half-written registry

    # ---------------- queries ----------------

    def list(self) -> List[Driver]:
        """Most recently used first, then by name."""
        return sorted(self._drivers.values(),
                      key=lambda d: (-(d.last_used_at or 0), d.name.lower()))

    def get(self, driver_id: str) -> Optional[Driver]:
        return self._drivers.get(driver_id)

    # ---------------- changes ----------------

    def create(self, raw_name: str) -> Driver:
        name = clean_name(raw_name)
        if any(d.name.lower() == name.lower() for d in self._drivers.values()):
            raise DriverNameError(f"'{name}' is already registered.")
        base = slugify(name)
        if base in _RESERVED:
            base = f"{base}-driver"
        driver_id, n = base, 2
        while driver_id in self._drivers:
            driver_id, n = f"{base}-{n}", n + 1
        d = Driver(id=driver_id, name=name)
        self._drivers[driver_id] = d
        self._save()
        return d

    def record_consent(self, driver_id: str) -> None:
        self._require(driver_id).consent_at = time.time()
        self._save()

    def record_calibration(self, driver_id: str, eye_signal: str, summary: dict) -> None:
        d = self._require(driver_id)
        d.eye_signal, d.calibration = eye_signal, dict(summary, at=time.time())
        self._save()

    def touch(self, driver_id: str) -> None:
        self._require(driver_id).last_used_at = time.time()
        self._save()

    def delete(self, driver_id: str) -> int:
        """Remove the driver, their eye profile and their session logs.
        Returns how many files were deleted."""
        self._require(driver_id)
        removed = 0
        profile = self.profiles_dir / f"{driver_id}.json"
        if profile.exists():
            profile.unlink()
            removed += 1
        if self.sessions_dir.exists():
            for f in self.sessions_dir.glob(f"*_{driver_id}_*.jsonl"):
                f.unlink()
                removed += 1
        del self._drivers[driver_id]
        self._save()
        return removed

    def _require(self, driver_id: str) -> Driver:
        d = self._drivers.get(driver_id)
        if d is None:
            raise KeyError(f"no driver {driver_id!r}")
        return d
