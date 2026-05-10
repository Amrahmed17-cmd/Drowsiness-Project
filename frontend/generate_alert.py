import struct, math, wave
from pathlib import Path

OUT = Path(__file__).parent.parent / "assets" / "alert.wav"
OUT.parent.mkdir(parents=True, exist_ok=True)

RATE     = 44100        # sample rate (Hz)
DURATION = 0.7          # seconds
FREQS    = [520, 680]   # two-tone alert
VOLUME   = 0.6

samples = []
n = int(RATE * DURATION)
for i in range(n):
    t = i / RATE
    fade = min(1.0, min(t / 0.02, (DURATION - t) / 0.05))  # attack + release envelope
    v = sum(math.sin(2 * math.pi * f * t) for f in FREQS) / len(FREQS)
    samples.append(int(v * fade * VOLUME * 32767))  # to int16 PCM

with wave.open(str(OUT), "w") as wf:
    wf.setnchannels(1)   # mono
    wf.setsampwidth(2)   # 16-bit
    wf.setframerate(RATE)
    wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

print(f"Alert sound written  {OUT}")
