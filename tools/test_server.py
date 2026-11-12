#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p "python3.withPackages (ps: [ ps.websockets ])"
"""Live test script for a running voicetserver (Voxtral + Qwen3-ASR).

Talks to the server over the network exactly like schmidispeech.user.js does:
HTTP API with the X-Api-Key header, WebSocket /asr?model=voxtral|qwen&api_key=…
with raw f32 LE PCM, 16 kHz mono, 4096-sample frames (256 ms, paced in real
time like the ScriptProcessor capture), the text frame "stop" at the end, and
{"type":"partial"|"final"|"error","text":…} replies. The "retranscribe" test
mimics the Diktate re-transcribe button instead: 1 s frames sent as fast as
possible, then "stop".

Run it BY HAND against a live server. Each test prints one PASS / FAIL / SKIP
/ INFO line with the reason, then a summary table. The exit code is 1 if any
test FAILed, else 0. INFO marks evidence a human has to judge (wording,
capitalisation at a pause boundary); read the printed finals.

---------------------------------------------------------------------------
On uplprodtest (NixOS; voicetserver listens on 127.0.0.1:8765 without TLS,
published on the tailnet by `tailscale serve` as svc:voicet over HTTPS)
---------------------------------------------------------------------------
  # the key is the agenix secret, readable by root / voicet only
  sudo ./tools/test_server.py --pair-id 0003          # key found automatically
  VOICET_API_KEY="$(sudo cat /run/agenix/voicet-apikey)" ./tools/test_server.py --wav s.wav
  # from another tailnet machine:
  ./tools/test_server.py --url https://voicet.<tailnet>.ts.net --api-key-file key.txt --wav s.wav

The shebang uses nix-shell (pulls python3 + websockets from the system's
nixpkgs); `python3 tools/test_server.py` works wherever `websockets` is
installed. Nothing else is needed (WAV handling is stdlib only).

API key lookup order: --api-key, $VOICET_API_KEY, --api-key-file,
/run/agenix/voicet-apikey, then api_key in the config file (--config-file,
default /var/lib/voicetserver/.config/voicetserver/config.toml, then
~/.config/voicetserver/config.toml). The key is never printed.

---------------------------------------------------------------------------
Audio input (no audio file ships with the repo — patient data)
---------------------------------------------------------------------------
Best: --pair-id ID streams one of your own calibration recordings straight
from the server (GET /training/audio/ID, read-only, kept in memory only). Its
calibration sentence becomes the --expect reference automatically. List the
IDs with --list-audio (also lists dictation-review candidates).
--review-id ID does the same with a saved real dictation
(GET /training/review/audio/ID); its stored text is the model transcript, not
a reference, so pass --expect yourself. Review candidates can contain patient
data: nothing is written to disk, but the finals are printed to the terminal.
On the server the files are 16-bit 16 kHz mono WAVs:
  /mnt/testupl/voicet/data/training/audio/NNNN.wav    (calibration pairs)
  /mnt/testupl/voicet/data/training/review/NNNN.wav   (saved dictations)
Or record a short German sentence (3-8 s, no long pause inside), e.g.
  arecord -f S16_LE -r 16000 -c 1 -d 6 satz.wav
Any PCM WAV works (8/16/24/32-bit int, any rate, any channel count): it is
downmixed and resampled to 16 kHz mono here.

---------------------------------------------------------------------------
Tests (--tests a,b,c; default: every non-destructive test)
---------------------------------------------------------------------------
  health       /health is "ready"; prints the open-connection count
  auth         /config and /asr without the key are rejected (401)
  config       /config version == --expect-version, model list
  pause        P1: the WAV twice with a synthesised pause in between, for each
               --pauses length (digital silence, optionally --noise-dbfs).
               Above the silence threshold: >= 2 finals, none empty or
               duplicated; with --expect also "the first word after the pause
               opens a new final". Capitalisation is INFO.
  wav          the WAV as it is, real time; finals printed (--min-finals N)
  tail         end of stream: speech right up to "stop", then the tail final
               must arrive; with --expect the last word must be in it
  retranscribe Diktate re-transcribe path (1 s frames, no pacing) with the
               longest pause; >= 2 finals
  idle         S1: send audio, then go quiet without closing; the server must
               close after ~30 s, and a new session on the engine must work
  idle-control 40 s of digital silence frames: the session must NOT be cut off
  lock         second Voxtral session is refused ("bereits … verwendet") while
               the first keeps working (Voxtral only)
  bogus        ?model=bogus is rejected over WS (error frame) and HTTP (400)
Per-engine tests run for --model voxtral|qwen|both (default both); qwen ones
SKIP when /config does not list "qwen". The qwen segment spacing fix (U2)
lives in the userscript and cannot be tested here.

---------------------------------------------------------------------------
Side effects of a default run
---------------------------------------------------------------------------
ASR sessions are NOT stored on the server: dictations reach the review pool
only through an explicit POST /training/review from the userscript, which
this script never calls. A default run writes nothing on the server apart
from log lines. BUT uplprodtest serves LIVE dictation and Voxtral allows one
session at a time: while a Voxtral test runs, a real dictation gets "Voxtral
wird bereits von einer anderen Sitzung verwendet". Run it when nobody
dictates (a warning is printed if /health shows open connections). A full
default run takes about 8-10 minutes.
"bogus" sends POST /lora/reload?model=bogus, which 0.1.38 rejects with 400
before doing anything; it is skipped unless /config reports --expect-version
or newer (an old server would reload the Voxtral LoRA instead).

---------------------------------------------------------------------------
--destructive (opt-in; these change server state and restore it)
---------------------------------------------------------------------------
  sentences    S5: POST /training/sentence adds a unique marker sentence, GET
               must list it as its own entry, DELETE removes it again.
               Restore: the DELETE. GET /training/sentences creates the stub
               file if it was missing (that file stays). If the add was glued
               onto the previous line (the S5 bug), the marker cannot be
               removed through the API: the script says so; delete the line
               "voicetserver-test-…" in data/training_sentences.txt by hand.
  qwen-switch  S2 (round trip only): /qwen/switch to the active slot (no-op),
               to the other slot, a qwen session, back to the original slot, a
               qwen session. Restore: switching back; the Qwen LoRA on/off
               state is put back if the switch changed it. Needs
               qwen_model_dir_alt. Qwen sessions of other users fail during
               each switch (seconds). The S2 failure case itself (switch back
               after a FAILED load) needs a broken slot and stays manual.
  config-save  S4: PATCH /config with an empty body ({}): no setting changes,
               but the server rewrites config.toml from its in-memory state
               (atomic temp file + rename). When the config file is readable
               here (run as root on the server) it checks mode 0600, that no
               config.toml.tmp is left and that every key is unchanged.
               Restore: nothing changed (formatting of the file may differ;
               the NixOS unit rewrites it on every start anyway).

---------------------------------------------------------------------------
Still manual (browser / userscript / server console)
---------------------------------------------------------------------------
U1 last syllable at the stop hotkey, U2 segment spacing in the text field, U3
config panel + text selection, U4 mic indicator after double-click / panel
close, U5 escaping, U6 pause compression without opening the panel, S1 with a
real network loss (Wi-Fi off / laptop sleep), S2 after a failed slot load, S3
'd' during model loading, P2 knock + long Qwen pause, P3 umlauts, P4
german_prime start, P5 capital after a custom-word replacement.
"""

import argparse
import array
import asyncio
import difflib
import json
import math
import os
import random
import re
import ssl
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

try:
    import tomllib
except ImportError:          # Python < 3.11: config-file parsing is skipped
    tomllib = None

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:          # reported in main(); pure helpers still importable
    websockets = None
    ConnectionClosed = Exception

SR = 16000                   # server sample rate
FRAME = 4096                 # userscript ScriptProcessor buffer (256 ms)
RETRANSCRIBE_FRAME = SR      # Diktate re-transcribe: 1 s per frame
VOXTRAL_TICK = 0.080         # Voxtral silence tick (s)
QWEN_TICK = 0.100            # Qwen silence tick (s)
WS_IDLE_TIMEOUT = 30.0       # server constant (src/main.rs WS_IDLE_TIMEOUT)
PARAGRAPH_OFFSET = 4         # server constant (paragraph_delay_offset)
BUSY_MARK = "bereits"        # Voxtral session-lock error text
DEFAULT_CONFIG_FILES = ["/var/lib/voicetserver/.config/voicetserver/config.toml",
                        os.path.expanduser("~/.config/voicetserver/config.toml")]
AGENIX_KEY = "/run/agenix/voicet-apikey"
NONDESTRUCTIVE = ["health", "auth", "config", "pause", "wav", "tail", "retranscribe",
                  "idle", "idle-control", "lock", "bogus"]
DESTRUCTIVE = ["sentences", "qwen-switch", "config-save"]
NEEDS_AUDIO = {"pause", "wav", "tail", "retranscribe", "idle", "lock", "qwen-switch"}


# ===========================================================================
# Pure helpers (audio, text) — unit-testable without a server
# ===========================================================================

def downmix(frames, channels):
    """Average interleaved channels to mono."""
    if channels == 1:
        return list(frames)
    return [sum(frames[i:i + channels]) / channels
            for i in range(0, len(frames) - channels + 1, channels)]


def resample_linear(samples, src_rate, dst_rate=SR):
    """Linear-interpolation resampler (adequate for speech at 16 kHz)."""
    if src_rate == dst_rate or not samples:
        return list(samples)
    n_out = int(round(len(samples) * dst_rate / src_rate))
    step = src_rate / dst_rate
    last = len(samples) - 1
    out = []
    for i in range(n_out):
        pos = i * step
        j = int(pos)
        if j >= last:
            out.append(samples[last])
        else:
            frac = pos - j
            out.append(samples[j] * (1.0 - frac) + samples[j + 1] * frac)
    return out


def decode_pcm_int(raw, sampwidth):
    """Little-endian integer PCM bytes → floats in [-1, 1)."""
    if sampwidth == 1:                                   # unsigned 8-bit
        return [(b - 128) / 128.0 for b in raw]
    if sampwidth == 2:
        a = array.array("h")
        a.frombytes(raw[:len(raw) // 2 * 2])
        if sys.byteorder == "big":
            a.byteswap()
        return [v / 32768.0 for v in a]
    if sampwidth == 3:
        out = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            out.append(v / 8388608.0)
        return out
    if sampwidth == 4:
        a = array.array("i")
        a.frombytes(raw[:len(raw) // 4 * 4])
        if sys.byteorder == "big":
            a.byteswap()
        return [v / 2147483648.0 for v in a]
    raise ValueError(f"unsupported sample width {sampwidth}")


def load_wav(source):
    """Read a PCM WAV (path or file object) → (16 kHz mono floats, info str)."""
    with wave.open(source, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    mono = downmix(decode_pcm_int(raw, width), ch)
    pcm = resample_linear(mono, rate, SR)
    info = f"{ch} ch, {width * 8}-bit, {rate} Hz, {len(pcm) / SR:.2f} s"
    if ch != 1 or rate != SR:
        info += " (converted to 16 kHz mono)"
    return pcm, info


def f32le(samples):
    """Floats → raw f32 little-endian bytes (the /asr wire format)."""
    a = array.array("f", samples)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def frame_rms(samples, frame):
    """RMS per consecutive frame of `frame` samples (last partial frame included)."""
    out = []
    for i in range(0, len(samples), frame):
        chunk = samples[i:i + frame]
        out.append(math.sqrt(sum(x * x for x in chunk) / len(chunk)) if chunk else 0.0)
    return out


def trim_silence(samples, threshold, frame=int(SR * VOXTRAL_TICK), keep_s=0.05):
    """Cut leading/trailing frames whose RMS is below `threshold`, keeping
    `keep_s` of margin, so a synthesised pause has exactly the intended length.
    Returns the samples unchanged when no frame is loud."""
    rms = frame_rms(samples, frame)
    loud = [i for i, r in enumerate(rms) if r >= threshold]
    if not loud:
        return list(samples)
    keep = int(keep_s * SR)
    start = max(0, loud[0] * frame - keep)
    end = min(len(samples), (loud[-1] + 1) * frame + keep)
    return list(samples[start:end])


def make_gap(seconds, noise_dbfs=None, seed=1):
    """`seconds` of digital silence, or Gaussian noise at `noise_dbfs` RMS."""
    n = int(round(seconds * SR))
    if noise_dbfs is None:
        return [0.0] * n
    sigma = 10 ** (noise_dbfs / 20.0)
    rnd = random.Random(seed)
    return [max(-1.0, min(1.0, rnd.gauss(0.0, sigma))) for _ in range(n)]


def synth_pause(speech, pause_s, noise_dbfs=None, tail_s=0.5):
    """speech + pause + speech + short tail. Returns (pcm, pause_start_s, pause_end_s)."""
    gap = make_gap(pause_s, noise_dbfs)
    pcm = list(speech) + gap + list(speech) + make_gap(tail_s, noise_dbfs, seed=2)
    start = len(speech) / SR
    return pcm, start, start + len(gap) / SR


_WORD_RE = re.compile(r"[\wÄÖÜäöüß'’-]+", re.UNICODE)


def split_words(text):
    """Words as written (case kept, punctuation dropped)."""
    return _WORD_RE.findall(text or "")


def norm_word(w):
    """Comparison form: lowercase, ß→ss (the server always writes ss)."""
    return w.lower().replace("ß", "ss").replace("’", "'")


def word_diff(expected, got):
    """Word-level diff of two texts. Returns a dict with wer, the opcode list
    (tag, expected words, got words), and explicit first/last word checks."""
    ew, gw = split_words(expected), split_words(got)
    en, gn = [norm_word(w) for w in ew], [norm_word(w) for w in gw]
    sm = difflib.SequenceMatcher(a=en, b=gn, autojunk=False)
    ops, errors = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            errors += max(i2 - i1, j2 - j1)
            ops.append((tag, ew[i1:i2], gw[j1:j2]))
    return {
        "wer": errors / len(ew) if ew else (0.0 if not gw else 1.0),
        "ops": ops,
        "first_ok": bool(en and gn and en[0] == gn[0]),
        "last_ok": bool(en and gn and en[-1] == gn[-1]),
        "first": (ew[0] if ew else None, gw[0] if gw else None),
        "last": (ew[-1] if ew else None, gw[-1] if gw else None),
    }


def format_diff(d):
    lines = [f"WER {d['wer'] * 100:.0f}%  "
             f"first word: expected {d['first'][0]!r} got {d['first'][1]!r} "
             f"({'ok' if d['first_ok'] else 'DIFF'}); "
             f"last word: expected {d['last'][0]!r} got {d['last'][1]!r} "
             f"({'ok' if d['last_ok'] else 'DIFF'})"]
    for tag, e, g in d["ops"]:
        lines.append(f"  {tag:8} expected [{' '.join(e)}] got [{' '.join(g)}]")
    return lines


def boundary_analysis(expected_copy, finals):
    """For a synthesised "copy + pause + copy" stream: locate the first word of
    the second copy (and the last word of the first) in the finals. Returns a
    dict, or None when the alignment cannot place the word."""
    exp = split_words(expected_copy)
    if not exp:
        return None
    exp2 = [norm_word(w) for w in exp + exp]
    hyp = []                                    # (word, final index, position in final)
    for fi, text in enumerate(finals):
        for pi, w in enumerate(split_words(text)):
            hyp.append((w, fi, pi))
    sm = difflib.SequenceMatcher(a=exp2, b=[norm_word(w) for w, _, _ in hyp], autojunk=False)
    amap = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            amap[blk.a + k] = blk.b + k
    k = len(exp)
    if k not in amap:
        return None
    w, fi, pi = hyp[amap[k]]
    res = {"word": w, "expected": exp[0], "final": fi, "pos": pi,
           "opens_final": pi == 0, "capitalised": w[:1].isupper(),
           "prev_last_closes": None}
    if k - 1 in amap:
        pw, pfi, ppi = hyp[amap[k - 1]]
        res["prev_last_closes"] = (pfi != fi
                                   and ppi == len(split_words(finals[pfi])) - 1)
    return res


def final_problems(finals, min_gap_s=None):
    """Empty finals and duplicated finals. `finals` = [(audio_s, text)].
    In a synthesised two-copy stream the same text may legitimately come
    twice; then only a repeat closer than `min_gap_s` of audio counts."""
    probs = []
    for i, (t, txt) in enumerate(finals):
        if not (txt or "").strip():
            probs.append(f"final #{i + 1} (at audio {t:.2f}s) is empty")
        if i and txt.strip() and txt.strip() == finals[i - 1][1].strip():
            gap = t - finals[i - 1][0]
            if min_gap_s is None or gap < min_gap_s:
                probs.append(f"final #{i + 1} duplicates #{i} ({txt.strip()!r}, {gap:.2f}s apart)")
    return probs


def version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return ()


# ===========================================================================
# Reporting
# ===========================================================================

class Reporter:
    def __init__(self):
        self.rows = []

    def add(self, name, status, reason):
        self.rows.append((name, status, reason))
        print(f"[{status:4}] {name}: {reason}", flush=True)

    def note(self, text):
        print(f"       {text}", flush=True)

    def summary(self):
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        w = max([len(n) for n, _, _ in self.rows] + [4])
        for n, s, r in self.rows:
            first = r if len(r) <= 70 else r[:67] + "..."
            print(f"{n:<{w}}  {s:<4}  {first}")
        counts = {}
        for _, s, _ in self.rows:
            counts[s] = counts.get(s, 0) + 1
        print("-" * 78)
        print("  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        return 1 if counts.get("FAIL") else 0


# ===========================================================================
# Server access
# ===========================================================================

class Server:
    def __init__(self, args, key):
        self.base = args.url.rstrip("/")
        self.key = key
        self.http_timeout = args.http_timeout
        self.ssl_ctx = None
        if self.base.startswith("https"):
            self.ssl_ctx = ssl.create_default_context()
            if args.insecure:
                self.ssl_ctx.check_hostname = False
                self.ssl_ctx.verify_mode = ssl.CERT_NONE
        u = urllib.parse.urlsplit(self.base)
        self.ws_base = ("wss" if u.scheme == "https" else "ws") + "://" + u.netloc + u.path + "/asr"

    def http(self, method, path, body=None, auth=True, timeout=None, raw=False):
        """→ (status, parsed JSON | text | bytes). Network errors → (None, message)."""
        data, headers = None, {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            headers["X-Api-Key"] = self.key
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.http_timeout,
                                        context=self.ssl_ctx) as r:
                status, payload = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, payload = e.code, e.read()
        except Exception as e:                          # timeout, refused, TLS
            return None, f"{type(e).__name__}: {e}"
        if raw:
            return status, payload
        txt = payload.decode("utf-8", "replace")
        try:
            return status, json.loads(txt)
        except ValueError:
            return status, txt

    def ws_url(self, model, with_key=True, extra=None):
        q = {}
        if model is not None:
            q["model"] = model
        if with_key:
            q["api_key"] = self.key
        q.update(extra or {})
        return self.ws_base + ("?" + urllib.parse.urlencode(q) if q else "")

    async def ws_open(self, model, with_key=True, timeout=15.0):
        # ping_interval=None: browsers send no WS pings, and a client ping
        # would count as traffic and defeat the server's 30 s idle timeout.
        kw = dict(open_timeout=timeout, ping_interval=None, close_timeout=5, max_size=None)
        if self.ws_base.startswith("wss"):
            kw["ssl"] = self.ssl_ctx
        return await asyncio.wait_for(
            websockets.connect(self.ws_url(model, with_key), **kw), timeout + 5)


class Event:
    __slots__ = ("t", "audio", "type", "text")

    def __init__(self, t, audio, typ, text):
        self.t, self.audio, self.type, self.text = t, audio, typ, text


class SessionResult:
    def __init__(self):
        self.events = []
        self.t0 = 0.0
        self.audio_sent = 0.0
        self.t_last_send = None
        self.t_stop = None
        self.t_closed = None
        self.timed_out = False
        self.close_info = ""
        self.connect_error = None

    @property
    def finals(self):
        return [e for e in self.events if e.type == "final"]

    @property
    def errors(self):
        return [e.text for e in self.events if e.type == "error"]

    @property
    def busy(self):
        return any(BUSY_MARK in (t or "") for t in self.errors)

    def text(self):
        return "".join(e.text or "" for e in self.finals).strip()


async def drive(ws, pcm, *, pace="realtime", chunk=FRAME, stop=True, close_wait=30.0):
    """Stream `pcm` over an open socket, optionally send "stop", then wait up
    to `close_wait` s for the server to close. Every wait is bounded."""
    res = SessionResult()
    res.t0 = time.monotonic()
    data = f32le(pcm)
    done = asyncio.Event()

    async def reader():
        try:
            async for msg in ws:
                now = time.monotonic() - res.t0
                if isinstance(msg, bytes):
                    continue
                try:
                    m = json.loads(msg)
                except ValueError:
                    m = {"type": "?", "text": msg}
                res.events.append(Event(now, res.audio_sent, m.get("type"), m.get("text", "")))
        except ConnectionClosed as e:
            res.close_info = f"closed ({getattr(e, 'code', None) or e})"
        except Exception as e:
            res.close_info = f"reader error {type(e).__name__}: {e}"
        finally:
            res.t_closed = time.monotonic() - res.t0
            done.set()

    rt = asyncio.create_task(reader())
    try:
        for off in range(0, len(pcm), chunk):
            if done.is_set() or res.errors:
                break
            n = min(chunk, len(pcm) - off)
            if pace == "realtime":                      # sent once "captured"
                delay = res.t0 + (off + n) / SR - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            try:
                await asyncio.wait_for(ws.send(data[off * 4:(off + n) * 4]), 10)
            except (ConnectionClosed, asyncio.TimeoutError):
                break
            res.audio_sent = (off + n) / SR
            res.t_last_send = time.monotonic() - res.t0
        if stop and not done.is_set():
            try:
                await asyncio.wait_for(ws.send("stop"), 10)
                res.t_stop = time.monotonic() - res.t0
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
        try:
            await asyncio.wait_for(done.wait(), close_wait)
        except asyncio.TimeoutError:
            res.timed_out = True
    finally:
        if not done.is_set():
            try:
                await asyncio.wait_for(ws.close(), 10)
            except Exception:
                pass
            try:
                await asyncio.wait_for(done.wait(), 5)
            except asyncio.TimeoutError:
                rt.cancel()
    return res


async def stream(srv, model, pcm, **kw):
    try:
        ws = await srv.ws_open(model)
    except Exception as e:
        res = SessionResult()
        res.connect_error = f"{type(e).__name__}: {e}"
        return res
    return await drive(ws, pcm, **kw)


def print_events(rep, res, show_partials=False, marks=None):
    """Print finals (and errors) with wall time and audio position at arrival."""
    for m in marks or []:
        rep.note(m)
    for e in res.events:
        if e.type == "partial" and not show_partials:
            continue
        rep.note(f"[{e.t:6.2f}s | audio {e.audio:6.2f}s] {e.type:7} {e.text!r}")
    if res.t_stop is not None:
        rep.note(f"[{res.t_stop:6.2f}s] stop sent; closed at "
                 f"{res.t_closed if res.t_closed is not None else float('nan'):.2f}s")


def session_trouble(res):
    """Common session failure as a reason string, or None."""
    if res.connect_error:
        return f"connect failed: {res.connect_error}"
    if res.busy:
        return ("Voxtral session lock held by another session (someone dictating, "
                "or a previous test session not yet released): " + res.errors[0])
    if res.errors:
        return "server error frame: " + res.errors[0]
    if res.timed_out:
        return "server did not close the session within the timeout after stop"
    return None


# ===========================================================================
# Tests
# ===========================================================================

class Ctx:
    def __init__(self, args, srv, rep):
        self.args, self.srv, self.rep = args, srv, rep
        self.cfg = {}
        self.ready = False
        self.speech = None          # trimmed speech (16 kHz floats)
        self.raw = None             # untrimmed WAV
        self.expect = args.expect
        self.models = []

    def threshold_s(self, model):
        s = int(self.cfg.get("silence_flush", 20))
        if model == "qwen":
            return s * QWEN_TICK
        return max(s, int(self.cfg.get("delay", 6)) + PARAGRAPH_OFFSET) * VOXTRAL_TICK

    def sil_thr(self):
        return float(self.cfg.get("silence_threshold", 0.006))


def t_health(c):
    st, body = c.srv.http("GET", "/health", auth=False)
    if st != 200 or not isinstance(body, dict):
        c.rep.add("health", "FAIL", f"GET /health → {st} {str(body)[:120]}")
        return
    conns = body.get("connections")
    if body.get("status") == "ready":
        c.ready = True
        c.rep.add("health", "PASS", f"status ready, {conns} open connection(s)")
    else:
        c.rep.add("health", "FAIL", f"status {body.get('status')!r} (loading or training) "
                                    "— ASR tests will be skipped")
    if conns:
        c.rep.note(f"WARNING: {conns} connection(s) open — someone may be dictating; "
                   "Voxtral tests then hit the single-session lock.")


async def t_auth(c):
    st, body = c.srv.http("GET", "/config", auth=False)
    ok_http = st == 401
    try:
        ws = await c.srv.ws_open("voxtral", with_key=False)
        await ws.close()
        ws_res = "upgrade accepted without key"
        ok_ws = False
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
        ok_ws = code == 401
        ws_res = f"upgrade rejected ({code or type(e).__name__})"
    c.rep.add("auth", "PASS" if ok_http and ok_ws else "FAIL",
              f"GET /config without key → {st}; /asr without key: {ws_res}")


def t_config(c):
    st, body = c.srv.http("GET", "/config")
    if st != 200 or not isinstance(body, dict):
        hint = " (wrong API key?)" if st == 401 else ""
        c.rep.add("config", "FAIL", f"GET /config → {st} {str(body)[:120]}{hint}")
        return False
    c.cfg = body
    ver, models = body.get("version"), body.get("models")
    ok = ver == c.args.expect_version and isinstance(models, list) and "voxtral" in models
    c.rep.add("config", "PASS" if ok else "FAIL",
              f"version {ver} (expected {c.args.expect_version}), models {models}, "
              f"server {body.get('server')!r}")
    c.rep.note(f"silence_flush {body.get('silence_flush')}, delay {body.get('delay')}, "
               f"min_speech {body.get('min_speech')}, silence_threshold "
               f"{body.get('silence_threshold')}, qwen slot {body.get('qwen_active_slot')} "
               f"(alt dir {'set' if body.get('qwen_model_dir_alt') else 'unset'})")
    c.rep.note(f"pause needed for a final: Voxtral ≈ {c.threshold_s('voxtral'):.2f}s, "
               f"Qwen ≈ {c.threshold_s('qwen'):.2f}s")
    return True


def model_gate(c, name, model):
    """SKIP reason for an engine test, or None."""
    if not c.ready:
        return "server not ready"
    if c.speech is None and name.split(":")[0] in NEEDS_AUDIO:
        return "no audio (give --wav, --pair-id or --review-id)"
    if model not in c.models:
        return (f"/config does not list '{model}' (engine disabled: no qwen_model_dir)"
                if model == "qwen" else f"model {model} not available")
    return None


async def t_pause(c, model):
    thr = c.threshold_s(model)
    margin = 0.25
    for p in c.args.pauses:
        name = f"pause:{model}:{p:g}s"
        if (why := model_gate(c, "pause", model)):
            c.rep.add(name, "SKIP", why)
            continue
        pcm, ps, pe = synth_pause(c.speech, p, c.args.noise_dbfs)
        res = await stream(c.srv, model, pcm, pace="realtime",
                           close_wait=30 + len(c.speech) / SR)
        if (why := session_trouble(res)):
            c.rep.add(name, "FAIL", why)
            print_events(c.rep, res)
            continue
        finals = res.finals
        texts = [e.text for e in finals]
        probs = final_problems([(e.audio, e.text) for e in finals],
                               min_gap_s=0.5 * len(c.speech) / SR)
        marks = [f"audio: speech 0–{ps:.2f}s, pause {ps:.2f}–{pe:.2f}s, speech {pe:.2f}–"
                 f"{pe + len(c.speech) / SR:.2f}s; threshold ≈ {thr:.2f}s"]
        above = p >= thr + margin
        below = p <= thr - margin
        verdict, why = "INFO", []
        if above:
            if len(finals) >= 2 and not probs:
                verdict = "PASS"
            else:
                verdict = "FAIL"
            why.append(f"{len(finals)} final(s) (need ≥2, pause above threshold)")
        elif below:
            pause_finals = [e for e in finals if ps < e.audio < pe + 0.3]
            why.append(f"{len(finals)} final(s); pause below threshold, "
                       f"{len(pause_finals)} final(s) arrived during the pause "
                       "(0 expected unless the WAV has its own pause)")
        else:
            why.append(f"{len(finals)} final(s); pause within ±{margin}s of the threshold")
        if probs:
            verdict = "FAIL"
            why += probs
        if c.expect:
            ba = boundary_analysis(c.expect, texts)
            if ba is None:
                why.append("boundary: first word after the pause not found in the finals")
            else:
                why.append(f"first word after pause {ba['word']!r} (expected {ba['expected']!r}) "
                           f"{'opens' if ba['opens_final'] else 'does NOT open'} final "
                           f"#{ba['final'] + 1}, {'capitalised' if ba['capitalised'] else 'lowercase'}")
                if above and not ba["opens_final"] and norm_word(ba["word"]) == norm_word(ba["expected"]):
                    verdict = "FAIL"
        c.rep.add(name, verdict, "; ".join(why))
        print_events(c.rep, res, c.args.show_partials, marks)
        if c.expect:
            for line in format_diff(word_diff(c.expect + " " + c.expect, res.text())):
                c.rep.note(line)


async def t_wav(c, model):
    name = f"wav:{model}"
    if (why := model_gate(c, "wav", model)):
        c.rep.add(name, "SKIP", why)
        return
    res = await stream(c.srv, model, c.raw, pace="realtime", close_wait=30 + len(c.raw) / SR)
    if (why := session_trouble(res)):
        c.rep.add(name, "FAIL", why)
        print_events(c.rep, res)
        return
    probs = final_problems([(e.audio, e.text) for e in res.finals])
    n = len(res.finals)
    if probs or (c.args.min_finals and n < c.args.min_finals):
        c.rep.add(name, "FAIL", f"{n} final(s) (min {c.args.min_finals}); " + "; ".join(probs))
    else:
        c.rep.add(name, "INFO", f"{n} final(s) — check wording and segment boundaries below")
    print_events(c.rep, res, c.args.show_partials)
    if c.expect:
        for line in format_diff(word_diff(c.expect, res.text())):
            c.rep.note(line)


async def t_tail(c, model):
    name = f"tail:{model}"
    if (why := model_gate(c, "tail", model)):
        c.rep.add(name, "SKIP", why)
        return
    pcm = trim_silence(c.speech, c.sil_thr(), keep_s=0.0)   # speech right up to "stop"
    res = await stream(c.srv, model, pcm, pace="realtime", close_wait=30)
    if (why := session_trouble(res)):
        c.rep.add(name, "FAIL", why)
        print_events(c.rep, res)
        return
    after = [e for e in res.events if res.t_stop is not None and e.t >= res.t_stop
             and e.type in ("final", "partial")]
    tail_finals = [e for e in after if e.type == "final" and e.text.strip()]
    if not tail_finals:
        c.rep.add(name, "FAIL", "no non-empty final after stop "
                                f"(after stop: {[(e.type, e.text) for e in after]})")
    elif c.expect:
        last = split_words(c.expect)[-1]
        got = [norm_word(w) for w in split_words(tail_finals[-1].text)]
        ok = norm_word(last) in got
        c.rep.add(name, "PASS" if ok else "FAIL",
                  f"tail final {tail_finals[-1].text!r} {'contains' if ok else 'LACKS'} "
                  f"the last reference word {last!r}; arrived "
                  f"{tail_finals[-1].t - res.t_stop:.2f}s after stop")
    else:
        c.rep.add(name, "PASS", f"tail final arrived {tail_finals[-1].t - res.t_stop:.2f}s "
                                f"after stop and the server closed; is the last word there? "
                                f"(INFO) {tail_finals[-1].text!r}")
    print_events(c.rep, res, c.args.show_partials)
    if c.expect:
        for line in format_diff(word_diff(c.expect, res.text())):
            c.rep.note(line)


async def t_retranscribe(c, model):
    name = f"retranscribe:{model}"
    if (why := model_gate(c, "retranscribe", model)):
        c.rep.add(name, "SKIP", why)
        return
    p = max(c.args.pauses)
    if p < c.threshold_s(model) + 0.25:
        c.rep.add(name, "SKIP", f"longest --pauses value {p}s is not above the threshold")
        return
    pcm, ps, pe = synth_pause(c.speech, p, c.args.noise_dbfs)
    dur = len(pcm) / SR
    res = await stream(c.srv, model, pcm, pace="burst", chunk=RETRANSCRIBE_FRAME,
                       close_wait=max(60.0, dur * 4 + 30))   # userscript's own timeout
    if (why := session_trouble(res)):
        c.rep.add(name, "FAIL", why)
        print_events(c.rep, res)
        return
    probs = final_problems([(e.audio, e.text) for e in res.finals], min_gap_s=0.0)
    n = len(res.finals)
    ok = n >= 2 and not probs
    c.rep.add(name, "PASS" if ok else "FAIL",
              f"{n} final(s) for {dur:.1f}s audio sent in burst with a {p:g}s pause"
              + ("; " + "; ".join(probs) if probs else ""))
    print_events(c.rep, res, c.args.show_partials,
                 [f"(burst mode: 'audio' column = audio already sent, not decoded)"])
    if c.expect:
        ba = boundary_analysis(c.expect, [e.text for e in res.finals])
        if ba:
            c.rep.note(f"first word after pause {ba['word']!r} opens a final: {ba['opens_final']}, "
                       f"capitalised: {ba['capitalised']}")
        for line in format_diff(word_diff(c.expect + " " + c.expect, res.text())):
            c.rep.note(line)


async def t_idle(c, model):
    name = f"idle:{model}"
    if (why := model_gate(c, "idle", model)):
        c.rep.add(name, "SKIP", why)
        return
    lo, hi = WS_IDLE_TIMEOUT - 3, WS_IDLE_TIMEOUT + c.args.idle_margin
    pcm = c.speech[:2 * SR]
    c.rep.note(f"{name}: 2 s of audio, then quiet on the wire (no close, no pings); "
               f"waiting up to {hi:.0f}s …")
    res = await stream(c.srv, model, pcm, pace="realtime", stop=False, close_wait=hi + 5)
    if (why := session_trouble(res)) and not res.timed_out:
        c.rep.add(name, "FAIL", why)
        return
    if res.timed_out or res.t_closed is None or res.t_last_send is None:
        c.rep.add(name, "FAIL", f"server did not close the idle session within {hi + 5:.0f}s")
        return
    idle = res.t_closed - res.t_last_send
    finals = [e.text for e in res.finals]
    ok = lo <= idle <= hi
    c.rep.add(name, "PASS" if ok else "FAIL",
              f"server closed after {idle:.1f}s of silence on the wire (expected "
              f"{lo:.0f}–{hi:.0f}s); final(s) before close: {finals}")
    # The lock / engine handle must be free again right away.
    res2 = await stream(c.srv, model, [0.0] * SR, pace="realtime", close_wait=15)
    why2 = session_trouble(res2)
    c.rep.add(name + ":reconnect", "FAIL" if why2 else "PASS",
              why2 or "new session on the same engine accepted right after the idle close")


async def t_idle_control(c, model):
    name = f"idle-control:{model}"
    if (why := model_gate(c, "idle-control", model)):
        c.rep.add(name, "SKIP", why)
        return
    secs = c.args.idle_control_seconds
    c.rep.note(f"{name}: sending {secs:g}s of digital-silence frames in real time …")
    res = await stream(c.srv, model, [0.0] * int(secs * SR), pace="realtime", close_wait=20)
    cut = res.t_closed is not None and res.t_stop is None
    why = session_trouble(res)
    if cut:
        c.rep.add(name, "FAIL", f"server closed the session after {res.t_closed:.1f}s although "
                                "frames kept coming" + (f" ({why})" if why else ""))
    elif why:
        c.rep.add(name, "FAIL", why)
    else:
        c.rep.add(name, "PASS", f"session survived {res.audio_sent:.0f}s of silence frames "
                                "and closed normally after stop")


async def t_lock(c):
    name = "lock:voxtral"
    if (why := model_gate(c, "lock", "voxtral")):
        c.rep.add(name, "SKIP", why)
        return
    try:
        a = await c.srv.ws_open("voxtral")
    except Exception as e:
        c.rep.add(name, "FAIL", f"session A connect failed: {e}")
        return
    res, b_msgs, b_err, refused = SessionResult(), [], None, False
    try:
        # The lock is taken right after the upgrade; one frame + a moment for the prefill.
        await asyncio.wait_for(a.send(f32le([0.0] * FRAME)), 10)
        await asyncio.sleep(1.5)
        try:
            b = await c.srv.ws_open("voxtral")
            try:
                while True:
                    m = await asyncio.wait_for(b.recv(), 10)
                    b_msgs.append(m)
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
            finally:
                try:
                    await asyncio.wait_for(b.close(), 5)
                except Exception:
                    pass
        except Exception as e:
            b_err = f"{type(e).__name__}: {e}"
        refused = any(BUSY_MARK in m for m in b_msgs if isinstance(m, str))
        res = await drive(a, c.speech + [0.0] * SR, pace="realtime",
                          close_wait=30 + len(c.speech) / SR)
    except (ConnectionClosed, asyncio.TimeoutError) as e:
        res.connect_error = f"session A lost: {type(e).__name__}"
    finally:
        try:
            await asyncio.wait_for(a.close(), 5)
        except Exception:
            pass
    why = session_trouble(res)
    a_ok = not why and bool(res.text())
    if res.busy:
        c.rep.add(name, "FAIL", "session A itself was refused — the engine was already busy: "
                                + res.errors[0])
        return
    c.rep.add(name, "PASS" if refused and a_ok else "FAIL",
              f"session B: {'refused with the documented error' if refused else 'NOT refused'} "
              f"({b_err or [m for m in b_msgs][:2]}); session A afterwards: "
              f"{'transcribed ' + repr(res.text()[:60]) if a_ok else (why or 'no text')}")


async def t_bogus(c):
    if not c.ready:
        c.rep.add("bogus:ws", "SKIP", "server not ready")
    else:
        res = await stream(c.srv, "bogus", [0.0] * FRAME, pace="burst", close_wait=10)
        ok = any("unknown model" in (t or "") for t in res.errors)
        c.rep.add("bogus:ws", "PASS" if ok else "FAIL",
                  f"/asr?model=bogus → {res.errors or res.connect_error or 'no error frame'}")
    if version_tuple(c.cfg.get("version")) < version_tuple(c.args.expect_version):
        c.rep.add("bogus:http", "SKIP", f"server version {c.cfg.get('version')} older than "
                                        f"{c.args.expect_version}: an old server might act on it")
        return
    st, body = c.srv.http("POST", "/lora/reload?model=bogus")
    ok = st == 400 and "unknown model" in str(body)
    c.rep.add("bogus:http", "PASS" if ok else "FAIL",
              f"POST /lora/reload?model=bogus → {st} {str(body)[:100]!r}")


# ---- destructive ----------------------------------------------------------

def t_sentences(c):
    name = "sentences"
    marker = f"voicetserver-test-{int(time.time())} Dies ist ein Testsatz."
    st, body = c.srv.http("GET", "/training/sentences")
    if st != 200:
        c.rep.add(name, "FAIL", f"GET /training/sentences → {st} {str(body)[:100]}")
        return
    before = [s.get("text") for s in body.get("sentences", [])]
    st, body = c.srv.http("POST", "/training/sentence", {"text": marker})
    if st != 200:
        c.rep.add(name, "FAIL", f"POST /training/sentence → {st} {str(body)[:100]} (nothing added)")
        return
    st, body = c.srv.http("GET", "/training/sentences")
    after = [s.get("text") for s in body.get("sentences", [])] if st == 200 else []
    listed = marker in after
    glued = [t for t in after if t and t != marker and t.endswith(marker)]
    st_d, _ = c.srv.http("DELETE", "/training/sentence", {"text": marker})
    st, body = c.srv.http("GET", "/training/sentences")
    final = [s.get("text") for s in body.get("sentences", [])] if st == 200 else None
    restored = final == before
    if listed and st_d == 200 and restored:
        c.rep.add(name, "PASS", "added sentence listed as its own entry; deleted again, "
                                "list identical to before")
    else:
        msg = f"listed: {listed}; delete → {st_d}; list restored: {restored}"
        if glued:
            msg += f"; GLUED onto a previous line: {glued[0]!r}"
        if not listed or st_d != 200:
            msg += ("; RESTORE BY HAND: remove the line containing "
                    f"'{marker.split()[0]}' from data/training_sentences.txt")
        c.rep.add(name, "FAIL", msg)


async def t_qwen_switch(c):
    name = "qwen-switch"
    if (why := model_gate(c, "qwen-switch", "qwen")):
        c.rep.add(name, "SKIP", why)
        return
    if not c.cfg.get("qwen_model_dir_alt"):
        c.rep.add(name, "SKIP", "qwen_model_dir_alt not configured (only one slot)")
        return
    orig = c.cfg.get("qwen_active_slot", "primary")
    other = "secondary" if orig == "primary" else "primary"
    lora0 = (c.cfg.get("lora_active_qwen"), c.cfg.get("lora_dir_qwen"))
    steps, ok = [], True
    sample = c.speech + [0.0] * (SR // 2)

    def switch(slot):
        t = time.monotonic()
        st, body = c.srv.http("POST", f"/qwen/switch?slot={slot}", timeout=c.args.switch_timeout)
        return st, body, time.monotonic() - t

    st, body, dt = switch(orig)
    steps.append(f"same slot {orig} → {st} in {dt:.1f}s")
    ok &= st == 200
    st, body, dt = switch(other)
    steps.append(f"→ {other}: {st} in {dt:.1f}s")
    if st == 200:
        res = await stream(c.srv, "qwen", sample, pace="realtime", close_wait=40)
        w = session_trouble(res)
        steps.append(f"qwen on {other}: {w or repr(res.text()[:50])}")
        ok &= not w and bool(res.text())
    else:
        ok = False
        steps.append(f"switch error {str(body)[:100]}")
    st, body, dt = switch(orig)
    steps.append(f"back → {orig}: {st} in {dt:.1f}s")
    back_ok = st == 200
    ok &= back_ok
    if back_ok:
        res = await stream(c.srv, "qwen", sample, pace="realtime", close_wait=40)
        w = session_trouble(res)
        steps.append(f"qwen on {orig}: {w or repr(res.text()[:50])}")
        ok &= not w and bool(res.text())
    # Restore check: slot and LoRA state as before.
    st, cfg = c.srv.http("GET", "/config")
    if st == 200 and isinstance(cfg, dict):
        lora1 = (cfg.get("lora_active_qwen"), cfg.get("lora_dir_qwen"))
        if lora1 != lora0:
            if lora0[0] is False and lora1[0]:
                rst, _ = c.srv.http("DELETE", "/lora?model=qwen")
            else:
                rst, _ = c.srv.http("POST", "/lora/reload?model=qwen", {"path": lora0[1]})
            steps.append(f"Qwen LoRA state changed {lora0} → {lora1}; restored → {rst}")
            st, cfg = c.srv.http("GET", "/config")
        slot_now = cfg.get("qwen_active_slot") if isinstance(cfg, dict) else None
        restored = slot_now == orig and isinstance(cfg, dict) and \
            (cfg.get("lora_active_qwen"), cfg.get("lora_dir_qwen")) == lora0
        steps.append(f"restored: slot {slot_now}, LoRA {'as before' if restored else 'DIFFERENT'}")
        ok &= restored
    else:
        ok = False
    if not back_ok:
        steps.append(f"RESTORE BY HAND: POST /qwen/switch?slot={orig}")
    c.rep.add(name, "PASS" if ok else "FAIL", "; ".join(steps))


def read_toml(path):
    if tomllib is None:
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def t_config_save(c):
    name = "config-save"
    path = c.args.config_file or next((p for p in DEFAULT_CONFIG_FILES if os.access(p, os.R_OK)), None)
    local = bool(path and os.access(path, os.R_OK))
    before = None
    if local:
        before = read_toml(path)
        m0 = os.stat(path).st_mtime_ns
    st, body = c.srv.http("PATCH", "/config", {})
    if st != 200:
        c.rep.add(name, "FAIL", f"PATCH /config {{}} → {st} {str(body)[:120]}")
        return
    same = isinstance(body, dict) and all(body.get(k) == c.cfg.get(k) for k in c.cfg)
    if not local:
        c.rep.add(name, "PASS" if same else "FAIL",
                  f"PATCH /config {{}} → 200, settings {'unchanged' if same else 'CHANGED'}; "
                  "config file not readable here — check mode 0600 / no .tmp by hand")
        return
    s = os.stat(path)
    tmp_left = os.path.exists(path + ".tmp")
    mode = stat.S_IMODE(s.st_mode)
    after = read_toml(path)
    keys_same = before == after if before is not None else None
    rewritten = s.st_mtime_ns != m0               # the server saves on every PATCH
    ok = same and rewritten and mode == 0o600 and not tmp_left and keys_same is not False
    c.rep.add(name, "PASS" if ok else "FAIL",
              f"PATCH {{}} → 200; file rewritten: {rewritten}; mode {oct(mode)}; "
              f".tmp left: {tmp_left}; keys unchanged: {keys_same}; API settings unchanged: {same}")


# ===========================================================================
# Main
# ===========================================================================

def find_key(args):
    if args.api_key:
        return args.api_key.strip(), "--api-key"
    if os.environ.get("VOICET_API_KEY"):
        return os.environ["VOICET_API_KEY"].strip(), "$VOICET_API_KEY"
    for p, label in ([(args.api_key_file, "--api-key-file")] if args.api_key_file else []) + \
            [(AGENIX_KEY, AGENIX_KEY)]:
        try:
            with open(p) as f:
                k = f.read().strip()
            if k:
                return k, label
        except OSError:
            pass
    for p in ([args.config_file] if args.config_file else []) + DEFAULT_CONFIG_FILES:
        try:
            k = (read_toml(p) or {}).get("api_key")
            if k:
                return k, p
        except (OSError, ValueError):
            pass
    return None, None


def load_audio(c):
    a = c.args
    if a.list_audio:
        for path, label in (("/training/pairs", "pairs"), ("/training/reviews", "reviews")):
            st, body = c.srv.http("GET", path)
            items = body.get(label, []) if isinstance(body, dict) else []
            print(f"{label} ({st}):")
            for it in items:
                txt = it.get("text", "")
                print(f"  {it.get('id')}  {it.get('duration_s', 0):5.1f}s  "
                      f"{txt[:60] if label == 'pairs' else '(dictation, text not shown)'}")
        sys.exit(0)
    data, src = None, None
    if a.wav:
        data, src = a.wav, a.wav
    elif a.pair_id or a.review_id:
        pid = a.pair_id or a.review_id
        path = f"/training/audio/{pid}" if a.pair_id else f"/training/review/audio/{pid}"
        st, raw = c.srv.http("GET", path, raw=True)
        if st != 200:
            print(f"cannot fetch {path}: {st}", file=sys.stderr)
            sys.exit(2)
        import io
        data, src = io.BytesIO(raw), f"server {path}"
        if a.pair_id and not c.expect:
            st, body = c.srv.http("GET", "/training/pairs")
            for it in (body.get("pairs", []) if isinstance(body, dict) else []):
                if it.get("id") == a.pair_id:
                    c.expect = it.get("text")
                    print(f"--expect taken from the calibration pair: {c.expect!r}")
    if data is None:
        return
    try:
        c.raw, info = load_wav(data)
    except (wave.Error, ValueError, OSError, EOFError) as e:
        print(f"cannot read WAV {src}: {e} (16-bit PCM WAV expected)", file=sys.stderr)
        sys.exit(2)
    c.speech = trim_silence(c.raw, c.sil_thr())
    print(f"audio: {src}: {info}; speech part {len(c.speech) / SR:.2f}s "
          f"(silence below RMS {c.sil_thr()} trimmed for the synthesised tests)")
    if c.args.noise_dbfs is not None and 10 ** (c.args.noise_dbfs / 20) >= c.sil_thr():
        print(f"WARNING: noise {c.args.noise_dbfs} dBFS is above silence_threshold — "
              "the pause will not count as silence")


async def amain(args):
    rep = Reporter()
    key, key_src = find_key(args)
    if not key:
        print("No API key: pass --api-key / $VOICET_API_KEY / --api-key-file, or run as root "
              f"on the server ({AGENIX_KEY}).", file=sys.stderr)
        return 2
    srv = Server(args, key)
    c = Ctx(args, srv, rep)
    print(f"server {srv.base} (WS {srv.ws_base}), API key from {key_src}")
    tests = [t.strip() for t in args.tests.split(",")] if args.tests else list(NONDESTRUCTIVE)
    if args.destructive and not args.tests:
        tests += DESTRUCTIVE
    unknown = [t for t in tests if t not in NONDESTRUCTIVE + DESTRUCTIVE]
    if unknown:
        print(f"unknown test(s): {unknown}", file=sys.stderr)
        return 2
    models = ["voxtral", "qwen"] if args.model == "both" else [args.model]

    t_health(c)                                   # always: gates the ASR tests
    if not t_config(c):                           # always: thresholds, model list
        return rep.summary()
    c.models = c.cfg.get("models") or ["voxtral"]
    load_audio(c)
    if "health" not in tests:
        rep.rows = [r for r in rep.rows if r[0] != "health"]
    if "config" not in tests:
        rep.rows = [r for r in rep.rows if r[0] != "config"]

    for t in tests:
        if t in DESTRUCTIVE and not args.destructive:
            rep.add(t, "SKIP", "destructive test: needs --destructive")
            continue
        if t == "auth":
            await t_auth(c)
        elif t == "pause":
            for m in models:
                await t_pause(c, m)
        elif t == "wav":
            for m in models:
                await t_wav(c, m)
        elif t == "tail":
            for m in models:
                await t_tail(c, m)
        elif t == "retranscribe":
            for m in models:
                await t_retranscribe(c, m)
        elif t == "idle":
            for m in models:
                await t_idle(c, m)
        elif t == "idle-control":
            for m in models:
                await t_idle_control(c, m)
        elif t == "lock":
            if "voxtral" in models:
                await t_lock(c)
        elif t == "bogus":
            await t_bogus(c)
        elif t == "sentences":
            t_sentences(c)
        elif t == "qwen-switch":
            await t_qwen_switch(c)
        elif t == "config-save":
            t_config_save(c)
    return rep.summary()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="test_server.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8765",
                   help="server base URL (default: %(default)s = voicetserver on uplprodtest "
                        "itself; via the tailnet: https://voicet.<tailnet>.ts.net)")
    p.add_argument("--api-key", help="API key (prefer $VOICET_API_KEY: argv is visible in ps)")
    p.add_argument("--api-key-file", help="file holding the API key")
    p.add_argument("--config-file", help="server config.toml (API key fallback; config-save checks)")
    p.add_argument("--insecure", action="store_true", help="do not verify the TLS certificate")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wav", help="speech WAV (PCM; resampled/downmixed to 16 kHz mono)")
    src.add_argument("--pair-id", help="stream calibration pair ID from the server (read-only)")
    src.add_argument("--review-id", help="stream saved dictation ID from the server (read-only)")
    src.add_argument("--list-audio", action="store_true", help="list pair/review IDs and exit")
    p.add_argument("--expect", help="reference text of the WAV (word diff, boundary checks)")
    p.add_argument("--model", choices=["voxtral", "qwen", "both"], default="both")
    p.add_argument("--tests", help="comma list: " + ",".join(NONDESTRUCTIVE + DESTRUCTIVE))
    p.add_argument("--destructive", action="store_true",
                   help="also run (or allow) " + ", ".join(DESTRUCTIVE) + " — they change and "
                        "then restore server state, see above")
    p.add_argument("--pauses", default="1.0,1.5,1.75,2.0,3.0",
                   help="pause lengths (s) for the pause test (default %(default)s)")
    p.add_argument("--noise-dbfs", type=float,
                   help="fill pauses with Gaussian noise at this RMS level (e.g. -60) "
                        "instead of digital silence")
    p.add_argument("--min-finals", type=int, default=0, help="wav test: require ≥ N finals")
    p.add_argument("--expect-version", default="0.1.38")
    p.add_argument("--show-partials", action="store_true")
    p.add_argument("--idle-margin", type=float, default=12.0,
                   help="idle test: accept a close up to 30 s + this (default %(default)s)")
    p.add_argument("--idle-control-seconds", type=float, default=40.0)
    p.add_argument("--http-timeout", type=float, default=15.0)
    p.add_argument("--switch-timeout", type=float, default=600.0,
                   help="qwen-switch: timeout per /qwen/switch call (default %(default)s s)")
    a = p.parse_args(argv)
    try:
        a.pauses = [float(x) for x in a.pauses.split(",") if x.strip()]
    except ValueError:
        p.error("--pauses: comma-separated seconds")
    if not a.pauses:
        p.error("--pauses is empty")
    return a


def main():
    args = parse_args()
    if websockets is None:
        print("python module 'websockets' missing: run via the nix-shell shebang "
              "(./tools/test_server.py) or install it", file=sys.stderr)
        return 2
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
