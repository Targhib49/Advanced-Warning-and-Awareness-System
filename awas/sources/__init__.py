"""Telemetry sources: mock, webcam, OBD-II, GPS, simulator.

GPS speed is the primary vehicle-signal path and OBD-II is an enhancement
(plan §7.4): heavy vehicles use SAE J1939, many older Indonesian vehicles
expose speed unreliably, and rail has no equivalent. Anything only OBD-II can
provide is Optional, and the system stays fully functional without it.
"""
