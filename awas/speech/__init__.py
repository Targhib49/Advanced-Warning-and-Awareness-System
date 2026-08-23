"""Speech I/O (plan §4.4).

Out: Piper TTS, offline, pre-warmed audio device.
In:  small offline STT for driver replies.

The reply is itself data: a driver who does not answer within 5 s is more
concerning than one who does, so response latency feeds back as a drowsiness
signal (plan §5.3).
"""
