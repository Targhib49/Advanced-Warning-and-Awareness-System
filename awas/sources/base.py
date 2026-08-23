"""Sensor contract. Real sensors and mocks both implement this."""

from abc import ABC, abstractmethod

from awas.core.telemetry import Telemetry


class Source(ABC):
    """A source of telemetry."""

    @abstractmethod
    def read(self) -> Telemetry:
        """Return the current reading as a Telemetry snapshot."""
        ...

    def close(self) -> None:
        """Release resources (camera handle, serial port). Override if needed."""
        pass
