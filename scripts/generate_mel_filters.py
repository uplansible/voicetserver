#!/usr/bin/env python3
"""Generate mel_filters.bin for voicetserver.

mel_filters.bin is a raw f32 LE binary of shape [128, 201] containing the
Whisper-style Slaney mel filterbank (N_FFT=400, N_MELS=128, SR=16kHz).
It is not included in the HuggingFace model repo and must be generated once.

Usage:
    python3 scripts/generate_mel_filters.py /path/to/model-dir
"""

import math
import struct
import sys
import os

SR     = 16000
N_FFT  = 400
N_MELS = 128
F_MIN  = 0.0
F_MAX  = SR / 2.0  # 8000 Hz

# Slaney mel scale (matches librosa htk=False, norm='slaney')
f_sp       = 200.0 / 3.0
min_log_hz = 1000.0
min_log_mel = (min_log_hz - F_MIN) / f_sp
logstep    = math.log(6.4) / 27.0

def hz_to_mel(f):
    if f < min_log_hz:
        return (f - F_MIN) / f_sp
    return min_log_mel + math.log(f / min_log_hz) / logstep

def mel_to_hz(m):
    if m < min_log_mel:
        return F_MIN + f_sp * m
    return min_log_hz * math.exp(logstep * (m - min_log_mel))

mel_min = hz_to_mel(F_MIN)
mel_max = hz_to_mel(F_MAX)
n_pts   = N_MELS + 2
mel_pts = [mel_min + i * (mel_max - mel_min) / (n_pts - 1) for i in range(n_pts)]
hz_pts  = [mel_to_hz(m) for m in mel_pts]

n_freqs  = N_FFT // 2 + 1  # 201
fft_freq = [i * SR / N_FFT for i in range(n_freqs)]

data = bytearray()
for m in range(1, N_MELS + 1):
    f_l = hz_pts[m - 1]
    f_c = hz_pts[m]
    f_r = hz_pts[m + 1]
    enorm = 2.0 / (f_r - f_l)  # Slaney area normalisation
    for f in fft_freq:
        v = max(0.0, min(
            (f - f_l) / (f_c - f_l) if f_c != f_l else 0.0,
            (f_r - f) / (f_r - f_c) if f_r != f_c else 0.0,
        )) * enorm
        data += struct.pack('<f', v)

expected = N_MELS * n_freqs * 4
assert len(data) == expected

out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
out_path = os.path.join(out_dir, 'mel_filters.bin')
with open(out_path, 'wb') as fh:
    fh.write(data)
print(f"Written {out_path} ({len(data)} bytes, {N_MELS}x{n_freqs} f32)")
