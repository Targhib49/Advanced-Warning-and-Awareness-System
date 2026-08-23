"""Main loop, scheduling, and session logging (plan §5.3, §5.4, §6).

Owns the things the state machine must not touch: the clock, devices, models,
and disk. Also owns the efficacy loop — recording response latency and
time-to-relapse after every intervention, in the format §6 will analyse.
"""
