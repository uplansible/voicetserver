# SCHMIDIspeech

Self-hosted real-time German medical dictation. Captures voice from the browser microphone,
transcribes it locally on a GPU server using **Voxtral Mini 4B Realtime** (via Candle),
and injects the result into the active text field on any website.

Connection between browser and server is secured via **Tailscale + TLS** — no public
internet exposure, no self-signed cert friction.

---

## Requirements

- GPU server: Ubuntu + NVIDIA GPU with CUDA 12.x
- Browser machine: on the same Tailscale tailnet as the GPU server
- Rust toolchain: stable 1.75+
- CUDA toolkit: matching your driver (see `docs/ubuntu_dependencies.md`)

---

## Quick-start variables

Every command below uses shell variables so you only fill in your values once.
Copy this block, edit the three lines, and paste it into your terminal before
running anything else. The variables stay set for the duration of your shell session.

```bash
# --- Set these once ---
SERVER_HOST="gpu-server.your-tailnet.ts.net"   # Tailscale FQDN of your GPU machine
SERVER_USER="youruser"                          # SSH / local user on the GPU server
MODEL_DIR="/path/to/models/Voxtral-Mini-4B-Realtime"  # Where the model files live
# ----------------------

# Derived — do not edit
CERT_DIR="/etc/tailscale/certs"
CERT="$CERT_DIR/$SERVER_HOST.crt"
KEY="$CERT_DIR/$SERVER_HOST.key"
WSS_URL="wss://$SERVER_HOST:8765/asr"
```

> **Why variables?**
> Every path and hostname appears in a dozen commands throughout this guide.
> Defining them once at the top means you change one line instead of hunting
> through every `scp`, `curl`, and `--tls-cert` flag when you move to a new
> server or tailnet. It also makes copy-pasting commands safe — no risk of
> accidentally leaving a placeholder like `gpu-server.your-tailnet.ts.net`
> in a real command.

---

## Build

### Dev (Docker / no GPU)

```bash
apt install -y pkg-config build-essential libssl-dev
cargo build
```

### Production (GPU server)

```bash
apt install -y pkg-config build-essential libssl-dev
# Install CUDA toolkit from NVIDIA repo — see docs/ubuntu_dependencies.md
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
# Replace 89 with your GPU's compute capability:
#   Ada Lovelace (RTX 4000/4090 etc) = 89
#   Ampere (RTX 3000 series)         = 86
#   Turing (RTX 2000 series)         = 75
```

Binary: `target/release/voicetserver`

---

## Model download

```bash
mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR"

HF_TOKEN="hf_your_token_here"   # huggingface.co/settings/tokens
BASE="https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/main"

wget --header="Authorization: Bearer $HF_TOKEN" "$BASE/tekken.json"
wget --header="Authorization: Bearer $HF_TOKEN" "$BASE/consolidated.safetensors"  # ~8.9 GB
```

`mel_filters.bin` is not in the HuggingFace repo. Copy the precomputed one from this repo:

```bash
cp assets/mel_filters.bin "$MODEL_DIR/"
```

Or regenerate it (pure Python 3, no dependencies):

```bash
python3 scripts/generate_mel_filters.py "$MODEL_DIR"
```

Expected directory:
```
$MODEL_DIR/
├── consolidated.safetensors   (~8.9 GB)
├── tekken.json
└── mel_filters.bin
```

---

## Tailscale TLS setup (one-time, on the GPU server)

Tailscale can provision a valid TLS certificate for your machine's FQDN at no cost,
so the browser trusts the `wss://` connection without any self-signed cert warnings.

**Step 1 — Enable HTTPS in the Tailscale admin console**

Go to [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines),
find your GPU server, click `…` → **Enable HTTPS**.

**Step 2 — Generate the cert on the GPU server**

```bash
sudo tailscale cert "$SERVER_HOST" && \
sudo mkdir -p "$CERT_DIR" && \
sudo mv "$SERVER_HOST".{crt,key} "$CERT_DIR/" && \
sudo chmod 600 "$KEY" && \
sudo chown "$SERVER_USER":"$SERVER_USER" "$CERT" "$KEY"
```

**Step 3 — Auto-renewal via cron (weekly)**

```bash
echo "0 3 * * 1 root tailscale cert $SERVER_HOST && chown $SERVER_USER:$SERVER_USER $CERT $KEY" \
  | sudo tee /etc/cron.d/tailscale-cert
```

---

## Start the server

**Development (no TLS, localhost only):**
```bash
./target/release/voicetserver --model-dir "$MODEL_DIR"
# Listens on ws://127.0.0.1:8765/asr
```

**Production (TLS, Tailscale):**
```bash
./target/release/voicetserver \
  --model-dir "$MODEL_DIR" \
  --bind-addr 0.0.0.0 \
  --tls-cert "$CERT" \
  --tls-key  "$KEY"
# Listens on wss://$SERVER_HOST:8765/asr
```

**Health check:**
```bash
curl https://"$SERVER_HOST":8765/health
# {"status":"ready","connections":0}
```

**Offline WAV test:**
```bash
./target/release/voicetserver --model-dir "$MODEL_DIR" audio.wav
```

---

## Browser userscript

1. Install [Violentmonkey](https://violentmonkey.github.io/) in Firefox or Chromium.
2. Open the Violentmonkey dashboard → **New script** → paste `schmidispeech.user.js`.
3. Set your server URL once (paste in the browser console on any page):
   ```javascript
   GM_setValue('server_url', 'wss://gpu-server.your-tailnet.ts.net:8765/asr');
   // Dev: GM_setValue('server_url', 'ws://127.0.0.1:8765/asr');
   ```
   You only need to do this once — Violentmonkey stores the value permanently.
4. A microphone button appears at the bottom-right of every page.
5. Click to dictate. Partial results appear in a tooltip; final text is injected
   into the active input field when a pause is detected.

**Audio format (Phase 1):** raw 16kHz mono f32 LE PCM over WebSocket.
Opus/WebM support planned for Phase 3.

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `.` | Directory with model files |
| `--device` | `0` | CUDA device index |
| `--delay` | `4` | Lookahead tokens (1–30; each = 80ms) |
| `--silence-threshold` | `0.006` | RMS silence threshold |
| `--silence-flush` | `20` | Consecutive silent chunks before paragraph break |
| `--min-speech` | `15` | Minimum speech chunks before silence arms |
| `--rms-ema` | `0.3` | EMA smoothing for RMS energy |
| `--port` | `8765` | WebSocket listen port |
| `--bind-addr` | `127.0.0.1` | Bind address (`0.0.0.0` for Tailscale) |
| `--tls-cert` | *(none)* | Path to TLS cert — enables `wss://` |
| `--tls-key` | *(none)* | Path to TLS private key |
| `--lora-adapter` | *(none)* | LoRA adapter dir (Phase 3, not yet wired) |

---

## Architecture

```
Browser mic (ScriptProcessorNode → 16kHz f32 PCM)
  │  wss://gpu-server:8765/asr   (Tailscale + TLS)
  ▼
voicetserver (Rust, headless, multi-connection)
  ├─ TLS: axum-server + rustls (Tailscale cert)
  ├─ Shared: Arc<VoxtralModel> — weights loaded once (~8.9 GB VRAM)
  ├─ GPU lock: tokio::sync::Mutex<ModelInner> — serial forward passes
  ├─ Per-connection: StreamingState { KV caches, SilenceDetector, mel buffer }
  ├─ Mel + Voxtral inference (Candle, BF16, Flash Attention 2)
  └─ WebSocket: {"type":"partial","text":"..."} / {"type":"final","text":"..."}
```

---

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Strip desktop GUI/input layer; dev build without CUDA |
| 1 | ✅ Done | WebSocket server + TLS, raw PCM audio, Violentmonkey userscript |
| 2 | Planned | Patient session vocabulary |
| 3 | Planned | Voice calibration, correction dictionary, LoRA fine-tuning |
| 4 | Planned | Macro expansion, review mode |
| 5 | Ideas | Scaling, cert renewal automation, AMD fallback |

See `TASKS.md` for detailed task tracking.
