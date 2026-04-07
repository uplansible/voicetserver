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
#   Ada Lovelake (RTX 4000/4090 etc) = 89
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

## Config file

All CLI flags can be set in `~/.config/voicetserver/config.toml` (created automatically
with a commented template on first run). CLI flags override config file values.

```toml
model_dir  = "/path/to/Voxtral-Mini-4B-Realtime"
bind_addr  = "0.0.0.0"
tls_cert   = "/etc/tailscale/certs/host.crt"
tls_key    = "/etc/tailscale/certs/host.key"
venv_path  = "/path/to/voicetserver-venv"   # Python venv for LoRA training

delay             = 6
silence_threshold = 0.006
silence_flush     = 20
min_speech        = 15
rms_ema           = 0.3
```

Runtime-adjustable without restart: `delay`, `silence_threshold`, `silence_flush`,
`min_speech`, `rms_ema` (via `PATCH /config` or the Server tab in the browser UI).

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
4. A microphone button appears at the bottom-right of every page.
5. Click (or press `Ctrl+Shift+D`) to dictate. Partial results appear in an overlay;
   final text is injected into the active input field when a pause is detected.

**Right-click the button** to open the settings panel:
- **Client tab** — server URL, hotkey
- **Server tab** — live inference parameters, custom word corrections
- **Training tab** — LoRA voice calibration (record sentences, trigger fine-tuning)

---

## LoRA voice calibration (Phase 2)

Fine-tune the decoder on your own voice and vocabulary without retraining the full model.

### 1. Install the Python venv (one-time, on the GPU server)

Put the venv on a partition with sufficient free space (torch + CUDA libs ≈ 5 GB).
`torch` must be installed from the PyTorch CUDA index; everything else comes from PyPI.

```bash
VENV="/path/to/voicetserver-venv"   # e.g. /mnt/data/voicetserver-venv
python3 -m venv $VENV
mkdir -p $VENV/tempdir

# Install torch for your CUDA version (replace cu128 with cu121/cu118 if needed)
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies from PyPI
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  safetensors mistral-common numpy tqdm packaging
```

Deploy the training script alongside the server binary:

```bash
mkdir -p ~/.local/bin/tools
cp tools/train_lora.py ~/.local/bin/tools/
```

Add to `~/.config/voicetserver/config.toml`:
```toml
venv_path = "/path/to/voicetserver-venv"
```

### 2. Add calibration sentences

Copy the sample sentences from the repo:
```bash
cp assets/training_sentences.txt ~/.config/voicetserver/training_sentences.txt
```
Edit the file to add your own domain-specific terms and phrases (one per line).

### 3. Collect training pairs

Open the **Training tab** in the browser UI:
- Browse through sentences with Prev/Next
- Click **Aufnehmen** to record yourself reading the sentence
- Edit the transcript if needed, then click **Speichern**
- Repeat for as many sentences as you like (20–50 pairs is a reasonable start)

Audio is saved to `~/.config/voicetserver/training/`.

### Training data guidelines

The LoRA adapts the model to your voice acoustics and domain vocabulary simultaneously.
Data quality and diversity matter more than volume.

**Sentences, not isolated words.**
Isolated words are pronounced artificially — slower, more deliberate, without natural
coarticulation. The model sees continuous speech at inference time, so train on it.

**Optimal sentence length: 6–12 words.**
- Too short (<5 words): unnatural prosody, minimal acoustic context
- Too long (>15 words): breath sounds, uneven pacing, recording fatigue
- ~8 words is the sweet spot — one breath, natural rhythm

**Use contextual variations for rare or problematic terms.**
Each variation teaches a different coarticulation context and sentence position.
For a term like *Miktionssituation*:
```
"Die Miktionssituation ist unauffällig."                    # end position
"Bezüglich der Miktionssituation keine Klagen."             # middle
"Miktionsbeschwerden wurden vom Patienten verneint."        # derived form
"Der Patient berichtet über eine normale Miktion."          # root form
"Postoperativ zeigte sich eine problemlose Miktion."        # prefix context
```
5–8 variations per target term is effective; beyond 10 the acoustic diversity saturates.

**Recommended dataset size: 60–100 pairs.**

| Pairs | Expected effect |
|-------|----------------|
| 10–20 | Minimal; slight acoustic adaptation |
| 30–60 | Noticeable improvement for targeted vocabulary |
| 60–100 | Good speaker + domain adaptation |
| 200+ | Diminishing returns; overfitting risk increases |

**One recording per sentence is enough.**
Re-recording the same sentence only helps if there is meaningful acoustic variation
(different energy, pacing). Diversity of *sentences* matters more than repetition.

**Speak at natural dictation pace** — not slow read-aloud mode.
The model will be used during real dictation; train it on the same speech style.

### 4. Train

Click **LoRA trainieren** in the Training tab (or `POST /training/run`).
The server runs `tools/train_lora.py` and streams progress to the log area.
The adapter is saved to `~/.config/voicetserver/lora_adapter/`.

### 5. Load the adapter

Add to `~/.config/voicetserver/config.toml` and restart the server:
```toml
lora_adapter = "/home/youruser/.config/voicetserver/lora_adapter"
```

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
| `--lora-adapter` | *(none)* | Path to LoRA adapter directory |
| `--venv-path` | *(none)* | Python venv for LoRA training |

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
  ├─ LoRA: optional decoder adapter (lora.rs) loaded at startup
  └─ WebSocket: {"type":"partial","text":"..."} / {"type":"final","text":"..."}

Training pipeline (POST /training/run):
  Browser Training tab → audio pairs → tools/train_lora.py (PyTorch)
  → adapter_model.safetensors → reload server with lora_adapter config
```

---

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Strip desktop GUI/input layer; dev build without CUDA |
| 1 | ✅ Done | WebSocket server + TLS, raw PCM audio, Violentmonkey userscript |
| 2 | ✅ Done | LoRA adapter loading, training pipeline, custom word corrections |
| 3 | Planned | Opus/WebM streaming, macro expansion |
| 4 | Planned | Patient session vocabulary |
| 5 | Ideas | Scaling, AMD fallback, automated cert renewal |

See `TASKS.md` for detailed task tracking.
