#!/usr/bin/env bash
# Installs voicetserver binary, LoRA venv, trainer scripts, and model files
# (Voxtral + optional Qwen3-ASR second engine); writes config.toml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$HOME/.config/voicetserver/config.toml"
DEFAULT_VENV="$HOME/.local/share/voicetserver-venv"
DEFAULT_MODEL_DIR="$HOME/models/Voxtral-Mini-4B-Realtime"
HF_BASE="https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/main"
DEFAULT_QWEN_DIR="$HOME/models/Qwen3-ASR-0.6B"
HF_QWEN_BASE="https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main"

echo "=== voicetserver installer ==="
echo ""

# --- Helper: write or update a key=value in config.toml ---
set_config_value() {
    local key="$1" val="$2"
    if grep -qE "^[#[:space:]]*${key}[[:space:]]*=" "$CONFIG_FILE"; then
        sed -i "s|^[#[:space:]]*${key}[[:space:]]*=.*|${key} = \"${val}\"|" "$CONFIG_FILE"
    else
        echo "${key} = \"${val}\"" >> "$CONFIG_FILE"
    fi
}

# --- Venv location ---
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^[[:space:]]*venv_path[[:space:]]*=' "$CONFIG_FILE" 2>/dev/null; then
    EXISTING_VENV=$(grep -E '^[[:space:]]*venv_path[[:space:]]*=' "$CONFIG_FILE" \
        | head -1 | sed 's/.*= *"\(.*\)"/\1/')
    [[ -n "$EXISTING_VENV" ]] && DEFAULT_VENV="$EXISTING_VENV"
fi
printf "Venv path [%s]: " "$DEFAULT_VENV"
read -r VENV_PATH
VENV_PATH="${VENV_PATH:-$DEFAULT_VENV}"

# --- Detect CUDA version → PyTorch index tag ---
detect_cuda_tag() {
    local ver=""
    if command -v nvcc &>/dev/null; then
        ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    fi
    if [[ -z "$ver" ]] && command -v nvidia-smi &>/dev/null; then
        ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    fi
    [[ -z "$ver" ]] && return
    local major="${ver%%.*}" minor="${ver##*.}"
    if   (( major > 12 || ( major == 12 && minor >= 8 ) )); then echo "cu128"
    elif (( major == 12 && minor >= 4 ));                   then echo "cu124"
    elif (( major == 12 && minor >= 1 ));                   then echo "cu121"
    elif (( major == 11 && minor >= 8 ));                   then echo "cu118"
    fi
}

CUDA_TAG=$(detect_cuda_tag || true)
if [[ -n "$CUDA_TAG" ]]; then
    echo "Detected CUDA → PyTorch index tag: $CUDA_TAG"
else
    echo "CUDA not detected. Available tags: cu128, cu124, cu121, cu118, cpu"
fi

printf "PyTorch CUDA index tag [%s]: " "${CUDA_TAG:-cpu}"
read -r USER_TAG
CUDA_TAG="${USER_TAG:-${CUDA_TAG:-cpu}}"

# --- Check python3 ---
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install it and re-run." >&2
    exit 1
fi

# --- Create venv ---
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.x")
echo ""
echo "Creating venv at: $VENV_PATH"
mkdir -p "$(dirname "$VENV_PATH")"
python3 -m venv "$VENV_PATH" || {
    echo "" >&2
    echo "Error: venv creation failed. On Debian/Ubuntu, install the venv package:" >&2
    echo "  sudo apt install python${PY_VER}-venv" >&2
    exit 1
}
mkdir -p "$VENV_PATH/tempdir"

# --- Install / upgrade torch ---
if "$VENV_PATH/bin/python3" -c "import torch" 2>/dev/null; then
    printf "torch already installed — upgrade? [y/N]: "
    read -r UPG_TORCH
    if [[ "${UPG_TORCH,,}" == "y" ]]; then
        echo "Upgrading torch from https://download.pytorch.org/whl/$CUDA_TAG ..."
        TMPDIR="$VENV_PATH/tempdir" "$VENV_PATH/bin/pip" install --no-cache-dir --upgrade \
            torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    fi
else
    echo "Installing torch from https://download.pytorch.org/whl/$CUDA_TAG ..."
    TMPDIR="$VENV_PATH/tempdir" "$VENV_PATH/bin/pip" install --no-cache-dir \
        torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# --- Install / upgrade remaining deps ---
if "$VENV_PATH/bin/python3" -c "import safetensors, mistral_common, tokenizers, numpy, tqdm, packaging" 2>/dev/null; then
    printf "Python deps already installed — upgrade? [y/N]: "
    read -r UPG_DEPS
    if [[ "${UPG_DEPS,,}" == "y" ]]; then
        echo "Upgrading dependencies..."
        TMPDIR="$VENV_PATH/tempdir" "$VENV_PATH/bin/pip" install --no-cache-dir --upgrade \
            safetensors mistral-common tokenizers numpy tqdm packaging
    fi
else
    echo "Installing remaining dependencies..."
    TMPDIR="$VENV_PATH/tempdir" "$VENV_PATH/bin/pip" install --no-cache-dir \
        safetensors mistral-common tokenizers numpy tqdm packaging
fi

# --- Deploy trainer scripts (per-model: voxtral + qwen) ---
# Server's find_script() checks ~/.config/voicetserver/tools/ as a universal fallback,
# so this location works regardless of where the binary lives.
DEPLOY_DIR="$HOME/.config/voicetserver/tools"
for TRAIN_NAME in train_lora_voxtral.py train_lora_qwen.py; do
    TRAIN_SCRIPT="$SCRIPT_DIR/$TRAIN_NAME"
    if [[ -f "$TRAIN_SCRIPT" ]]; then
        mkdir -p "$DEPLOY_DIR"
        cp "$TRAIN_SCRIPT" "$DEPLOY_DIR/"
        echo "Deployed $TRAIN_NAME → $DEPLOY_DIR/"
    else
        echo "Warning: $TRAIN_NAME not found at $TRAIN_SCRIPT — deploy manually."
    fi
done

# --- Ensure config dir + file exist ---
mkdir -p "$(dirname "$CONFIG_FILE")"
if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" <<TOML
# voicetserver configuration
# Restart required for: model_dir, qwen_model_dir, language, device, port, bind_addr, tls_cert, tls_key, lora_adapter, lora_adapter_qwen, venv_path
# Runtime-adjustable via PATCH /config: delay, silence_threshold, silence_flush, min_speech, rms_ema, fuzzy_hotwords, fuzzy_max_ratio, german_prime, context_biasing

bind_addr = "127.0.0.1"
port = 8765
TOML
    echo "Created $CONFIG_FILE"
fi

set_config_value "venv_path" "$VENV_PATH"
echo "venv_path set in $CONFIG_FILE"

# --- Data directory ---
DEFAULT_DATA_DIR="$HOME/.config/voicetserver"
EXISTING_DATA_DIR=""
if grep -qE '^[[:space:]]*data_dir[[:space:]]*=' "$CONFIG_FILE" 2>/dev/null; then
    EXISTING_DATA_DIR=$(grep -E '^[[:space:]]*data_dir[[:space:]]*=' "$CONFIG_FILE" \
        | head -1 | sed 's/.*= *"\(.*\)"/\1/')
fi

DATA_DIR=""
if [[ -n "$EXISTING_DATA_DIR" ]]; then
    printf "Data dir currently %s — keep this location? [Y/n]: " "$EXISTING_DATA_DIR"
    read -r KEEP_DATA
    if [[ "${KEEP_DATA,,}" == "n" ]]; then
        printf "New data directory [%s]: " "$EXISTING_DATA_DIR"
        read -r DATA_DIR
        DATA_DIR="${DATA_DIR:-$EXISTING_DATA_DIR}"
        set_config_value "data_dir" "$DATA_DIR"
        echo "data_dir updated to: $DATA_DIR"
    else
        DATA_DIR="$EXISTING_DATA_DIR"
    fi
else
    printf "Data directory for custom_words, training, LoRA [%s]: " "$DEFAULT_DATA_DIR"
    read -r DATA_DIR
    DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
    if [[ "$DATA_DIR" != "$DEFAULT_DATA_DIR" ]]; then
        set_config_value "data_dir" "$DATA_DIR"
        echo "data_dir set to: $DATA_DIR"
    fi
fi

# --- Voxtral model files ---
echo ""

# Read existing model_dir from config if present
EXISTING_MODEL_DIR=""
if grep -qE '^[[:space:]]*model_dir[[:space:]]*=' "$CONFIG_FILE" 2>/dev/null; then
    EXISTING_MODEL_DIR=$(grep -E '^[[:space:]]*model_dir[[:space:]]*=' "$CONFIG_FILE" \
        | head -1 | sed 's/.*= *"\(.*\)"/\1/')
fi
[[ -n "$EXISTING_MODEL_DIR" ]] && DEFAULT_MODEL_DIR="$EXISTING_MODEL_DIR"

MODEL_DIR=""
if [[ -n "$EXISTING_MODEL_DIR" && -f "$EXISTING_MODEL_DIR/consolidated.safetensors" ]]; then
    printf "Model already at %s — keep this location? [Y/n]: " "$EXISTING_MODEL_DIR"
    read -r KEEP_MODEL
    if [[ "${KEEP_MODEL,,}" == "n" ]]; then
        printf "New model directory [%s]: " "$EXISTING_MODEL_DIR"
        read -r MODEL_DIR
        MODEL_DIR="${MODEL_DIR:-$EXISTING_MODEL_DIR}"
        set_config_value "model_dir" "$MODEL_DIR"
        echo "model_dir updated to: $MODEL_DIR"
        echo "Note: move model files from $EXISTING_MODEL_DIR to $MODEL_DIR, or re-download."
    else
        MODEL_DIR="$EXISTING_MODEL_DIR"
    fi
else
    printf "Download Voxtral-Mini-4B-Realtime model files (~8.9 GB)? [Y/n]: "
    read -r DL_CHOICE
    if [[ "${DL_CHOICE,,}" != "n" ]]; then
        printf "Model directory [%s]: " "$DEFAULT_MODEL_DIR"
        read -r MODEL_DIR
        MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
        mkdir -p "$MODEL_DIR"
        echo "Downloading model files to: $MODEL_DIR"
        HF_FILES=(tekken.json consolidated.safetensors)
        for f in "${HF_FILES[@]}"; do
            if [[ -f "$MODEL_DIR/$f" ]]; then
                echo "  $f — already present, skipping"
            else
                echo "  Downloading $f ..."
                if ! wget -q --show-progress -O "$MODEL_DIR/$f" "$HF_BASE/$f"; then
                    rm -f "$MODEL_DIR/$f"
                    echo "  Warning: $f not available from HuggingFace" >&2
                fi
            fi
        done
    else
        printf "Model directory (will be written to config) [%s]: " "$DEFAULT_MODEL_DIR"
        read -r MODEL_DIR
        MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
    fi
    set_config_value "model_dir" "$MODEL_DIR"
    echo "model_dir set to: $MODEL_DIR"
fi

# --- Copy precomputed mel filterbank into the model dir if missing ---
# mel_filters.bin is not published on HuggingFace; it ships in the repo under assets/.
if [[ -n "$MODEL_DIR" && ! -f "$MODEL_DIR/mel_filters.bin" ]]; then
    if [[ -f "$REPO_ROOT/assets/mel_filters.bin" ]]; then
        cp "$REPO_ROOT/assets/mel_filters.bin" "$MODEL_DIR/"
        echo "Copied mel_filters.bin → $MODEL_DIR/"
    else
        echo "Warning: assets/mel_filters.bin not found — copy or regenerate it manually:" >&2
        echo "  python3 scripts/generate_mel_filters.py $MODEL_DIR" >&2
    fi
fi

# --- Qwen3 model files (optional second engine) ---
echo ""

# Read existing qwen_model_dir from config if present (set_config_value uncomments
# the templated "# qwen_model_dir = ..." line in place, so only match active keys)
EXISTING_QWEN_DIR=""
if grep -qE '^[[:space:]]*qwen_model_dir[[:space:]]*=' "$CONFIG_FILE" 2>/dev/null; then
    EXISTING_QWEN_DIR=$(grep -E '^[[:space:]]*qwen_model_dir[[:space:]]*=' "$CONFIG_FILE" \
        | head -1 | sed 's/.*= *"\(.*\)"/\1/')
fi
[[ -n "$EXISTING_QWEN_DIR" ]] && DEFAULT_QWEN_DIR="$EXISTING_QWEN_DIR"

QWEN_DIR=""
if [[ -n "$EXISTING_QWEN_DIR" && -f "$EXISTING_QWEN_DIR/model.safetensors" ]]; then
    printf "Qwen3 model already at %s — keep this location? [Y/n]: " "$EXISTING_QWEN_DIR"
    read -r KEEP_QWEN
    if [[ "${KEEP_QWEN,,}" == "n" ]]; then
        printf "New Qwen3 model directory [%s]: " "$EXISTING_QWEN_DIR"
        read -r QWEN_DIR
        QWEN_DIR="${QWEN_DIR:-$EXISTING_QWEN_DIR}"
        set_config_value "qwen_model_dir" "$QWEN_DIR"
        echo "qwen_model_dir updated to: $QWEN_DIR"
        echo "Note: move model files from $EXISTING_QWEN_DIR to $QWEN_DIR, or re-download."
    else
        QWEN_DIR="$EXISTING_QWEN_DIR"
    fi
else
    printf "Enable the Qwen3-ASR second engine (download ~1.8 GB)? [Y/n]: "
    read -r QWEN_CHOICE
    if [[ "${QWEN_CHOICE,,}" != "n" ]]; then
        printf "Qwen3 model directory [%s]: " "$DEFAULT_QWEN_DIR"
        read -r QWEN_DIR
        QWEN_DIR="${QWEN_DIR:-$DEFAULT_QWEN_DIR}"
        mkdir -p "$QWEN_DIR"
        echo "Downloading Qwen3 model files to: $QWEN_DIR"
        QWEN_FILES=(config.json tokenizer.json tokenizer_config.json vocab.json
                    merges.txt preprocessor_config.json generation_config.json
                    model.safetensors)
        for f in "${QWEN_FILES[@]}"; do
            if [[ -f "$QWEN_DIR/$f" ]]; then
                echo "  $f — already present, skipping"
            else
                echo "  Downloading $f ..."
                if ! wget -q --show-progress -O "$QWEN_DIR/$f" "$HF_QWEN_BASE/$f"; then
                    rm -f "$QWEN_DIR/$f"
                    echo "  Warning: $f not available from HuggingFace" >&2
                fi
            fi
        done
        set_config_value "qwen_model_dir" "$QWEN_DIR"
        echo "qwen_model_dir set to: $QWEN_DIR"
    else
        echo "Skipped — qwen engine disabled (set qwen_model_dir in config.toml to enable later)."
    fi
fi

# --- Generate tokenizer.json if missing ---
# tokenizer.json is not published on HuggingFace; derive it from the tokenizer
# config using transformers (installed into the venv on demand — the trainers
# themselves only need the lighter `tokenizers` package).
if [[ -n "$QWEN_DIR" && -f "$QWEN_DIR/model.safetensors" && ! -f "$QWEN_DIR/tokenizer.json" ]]; then
    echo ""
    echo "tokenizer.json not found — generating from tokenizer config ..."
    if ! "$VENV_PATH/bin/python3" -c "import transformers" 2>/dev/null; then
        echo "Installing transformers into the venv (needed once for tokenizer generation) ..."
        TMPDIR="$VENV_PATH/tempdir" "$VENV_PATH/bin/pip" install --no-cache-dir transformers
    fi
    QWEN_DIR="$QWEN_DIR" "$VENV_PATH/bin/python3" -c "
import os
model_dir = os.environ['QWEN_DIR']
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tok.save_pretrained(model_dir)
print('  tokenizer.json written to', model_dir)
" || echo "  Warning: could not generate tokenizer.json — see CLAUDE.md for the manual command." >&2
fi

# --- Install voicetserver binary ---
echo ""
PREBUILT_CPU="$SCRIPT_DIR/voicetserver"
PREBUILT_GPU="$SCRIPT_DIR/voicetserver-cuda"
COMPILED="$REPO_ROOT/target/release/voicetserver"

CARGO_VER=$(grep '^version' "$REPO_ROOT/Cargo.toml" | head -1 | grep -oP '".*?"' | tr -d '"')
CPU_VER="" ; GPU_VER=""
[[ -f "$PREBUILT_CPU" ]] && CPU_VER=$("$PREBUILT_CPU" --version 2>/dev/null | awk '{print $2}') || true
[[ -f "$PREBUILT_GPU" ]] && GPU_VER=$("$PREBUILT_GPU" --version 2>/dev/null | awk '{print $2}') || true

# Build option list dynamically
OPTIONS=()
[[ -f "$PREBUILT_CPU" ]] && OPTIONS+=("cpu")
[[ -f "$PREBUILT_GPU" ]] && OPTIONS+=("gpu")
OPTIONS+=("compile")

echo "Available binaries:"
[[ -f "$PREBUILT_CPU" ]] && echo "  cpu     — prebuilt CPU-only     v${CPU_VER:-?}"
[[ -f "$PREBUILT_GPU" ]] && echo "  gpu     — prebuilt CUDA (GPU)    v${GPU_VER:-?}"
echo "  compile — build from source      v${CARGO_VER}"

DEFAULT_BUILD="${OPTIONS[0]}"
printf "Which to install? [%s, default: %s]: " "$(IFS=/; echo "${OPTIONS[*]}")" "$DEFAULT_BUILD"
read -r BUILD_CHOICE
BUILD_CHOICE="${BUILD_CHOICE:-$DEFAULT_BUILD}"

BINARY_TO_INSTALL=""
if [[ "$BUILD_CHOICE" == "compile" ]]; then
    if [[ ! -f "$COMPILED" ]]; then
        # --- Ask about CUDA (only if a GPU is present) ---
        COMPUTE_CAP=""
        HAS_CUDA=false
        if command -v nvidia-smi &>/dev/null; then
            COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d '.')
            [[ -n "$COMPUTE_CAP" ]] && HAS_CUDA=true
        fi
        USE_CUDA="n"
        if [[ "$HAS_CUDA" == true ]]; then
            printf "Build with CUDA? [y/N]: "
            read -r USE_CUDA
        else
            echo "No GPU detected — building voicetserver (CPU) ..."
        fi
        if [[ "${USE_CUDA,,}" == "y" ]]; then
            printf "Compute capability (e.g. 89 for RTX 4090) [%s]: " "${COMPUTE_CAP:-89}"
            read -r USER_CAP
            COMPUTE_CAP="${USER_CAP:-${COMPUTE_CAP:-89}}"
            echo "Building voicetserver (CUDA, compute cap $COMPUTE_CAP) ..."
            if ! (cd "$REPO_ROOT" && CUDA_PATH=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" \
                    CUDA_COMPUTE_CAP="$COMPUTE_CAP" cargo build --release --features cuda); then
                echo "Error: cargo build failed — see output above." >&2
                exit 1
            fi
        else
            echo "Building voicetserver (CPU) ..."
            if ! (cd "$REPO_ROOT" && cargo build --release); then
                echo "Error: cargo build failed — see output above." >&2
                exit 1
            fi
        fi
    fi
    BINARY_TO_INSTALL="$COMPILED"
elif [[ "$BUILD_CHOICE" == "gpu" ]]; then
    BINARY_TO_INSTALL="$PREBUILT_GPU"
else
    BINARY_TO_INSTALL="$PREBUILT_CPU"
fi

if [[ -f "$BINARY_TO_INSTALL" ]]; then
    mkdir -p "$HOME/.local/bin"
    install -m 755 "$BINARY_TO_INSTALL" "$HOME/.local/bin/voicetserver"
    echo "Installed binary → ~/.local/bin/voicetserver"
else
    echo "Warning: binary not found at $BINARY_TO_INSTALL — install manually."
fi

# --- Ensure ~/.local/bin is in PATH ---
ensure_path() {
    local profile_file=""
    if [[ -n "${ZSH_VERSION:-}" || "$SHELL" == */zsh ]]; then
        profile_file="$HOME/.zshrc"
    else
        profile_file="$HOME/.bashrc"
    fi
    local line='export PATH="$HOME/.local/bin:$PATH"'
    if ! grep -qF '.local/bin' "$profile_file" 2>/dev/null; then
        echo "" >> "$profile_file"
        echo "# added by voicetserver installer" >> "$profile_file"
        echo "$line" >> "$profile_file"
        echo "Added ~/.local/bin to PATH in $profile_file"
        echo "  Run: source $profile_file  (or open a new terminal)"
    fi
    export PATH="$HOME/.local/bin:$PATH"
}

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    ensure_path
else
    echo "~/.local/bin already in PATH"
fi

# --- Exposure: Tailscale Service (recommended) or self-hosted TLS ---
#
# Two ways to reach the server from another machine:
#
#   [1] Tailscale Service — tailscaled terminates TLS on a per-service virtual IP and
#       proxies to 127.0.0.1. The server speaks plain HTTP and is not reachable from the
#       LAN at all. No cert to renew: an expired cert can no longer kill dictation.
#   [2] Self-hosted TLS   — the server holds a `tailscale cert` for the node FQDN and binds
#       0.0.0.0. This is what earlier versions of this installer did; it works, but it
#       exposes the port to the whole LAN and needs a weekly renewal timer.
#
# Mode [1] needs the *service* to exist before a node can advertise it — see the prompt.
echo ""

# Comment a key out rather than deleting it, so the previous value stays visible.
unset_config_value() {
    local key="$1"
    [[ -f "$CONFIG_FILE" ]] || return 0
    sed -i "s|^[[:space:]]*${key}[[:space:]]*=|# ${key} =|" "$CONFIG_FILE"
}

# Port the server listens on — config wins, else the compiled default.
SERVICE_PORT=8765
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^[[:space:]]*port[[:space:]]*=' "$CONFIG_FILE" 2>/dev/null; then
    CFG_PORT=$(grep -E '^[[:space:]]*port[[:space:]]*=' "$CONFIG_FILE" | head -1 \
        | grep -oE '[0-9]+' | head -1)
    [[ -n "$CFG_PORT" ]] && SERVICE_PORT="$CFG_PORT"
fi

CERT_CONFIGURED=false
if command -v tailscale &>/dev/null; then
    TS_HOST=$(tailscale status --json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" \
        2>/dev/null || true)

    echo "How should voicetserver be reachable from other machines?"
    echo "  [1] Tailscale Service — TLS at the proxy, server binds 127.0.0.1   (recommended)"
    echo "  [2] Self-hosted TLS   — server holds the cert, binds 0.0.0.0       (legacy)"
    echo "  [3] Skip — configure exposure later"
    printf "Choice [1]: "
    read -r EXPOSURE_CHOICE
    EXPOSURE_CHOICE="${EXPOSURE_CHOICE:-1}"

    if [[ "$EXPOSURE_CHOICE" == "1" ]]; then
        printf "Service name (without the svc: prefix) [voicet]: "
        read -r SVC_NAME
        SVC_NAME="${SVC_NAME:-voicet}"

        echo ""
        echo "Two things must already be true — neither can be done from here:"
        echo "  1. This machine carries an ACL tag. A service host cannot be an owned node;"
        echo "     tag it in the admin console under Machine -> Edit ACL tags."
        echo "  2. The service exists: Services -> Advertise -> Define a Service,"
        echo "     name '${SVC_NAME}', endpoint 'tcp:443', tag field left empty."
        echo "     'tailscale serve --service=' does NOT create it. Advertising a service"
        echo "     that was never defined writes local config, reports that approval is"
        echo "     required, and shows up nowhere in the console."
        printf "Both done? [y/N]: "
        read -r SVC_READY

        if [[ "${SVC_READY,,}" == "y" ]]; then
            set_config_value "bind_addr" "127.0.0.1"
            # Must go: a proxy sending plain HTTP into a TLS listener answers 502.
            unset_config_value "tls_cert"
            unset_config_value "tls_key"
            echo "Config updated: bind_addr=127.0.0.1, tls_cert/tls_key disabled"

            if sudo tailscale serve --service="svc:${SVC_NAME}" --https=443 \
                    "http://127.0.0.1:${SERVICE_PORT}"; then
                CERT_CONFIGURED=true
                echo ""
                echo "Approve the host: Services -> ${SVC_NAME} -> Service hosts -> Approve."
                echo "Until then the address does not resolve to anything and a browser"
                echo "reports that it cannot establish a secure connection."
                echo ""
                echo "If it still reports 'approval from an admin is required' after you"
                echo "approved it, the daemon is holding the old state:"
                echo "  tailscale serve clear svc:${SVC_NAME}"
                echo "  sleep 2"
                echo "  sudo tailscale serve --service=svc:${SVC_NAME} --https=443 http://127.0.0.1:${SERVICE_PORT}"
            else
                echo "Warning: advertising the service failed — run it manually:" >&2
                echo "  sudo tailscale serve --service=svc:${SVC_NAME} --https=443 http://127.0.0.1:${SERVICE_PORT}" >&2
            fi
        else
            echo "Skipped. Once both are done:"
            echo "  sudo tailscale serve --service=svc:${SVC_NAME} --https=443 http://127.0.0.1:${SERVICE_PORT}"
            echo "and set bind_addr = \"127.0.0.1\" with tls_cert/tls_key removed in:"
            echo "  $CONFIG_FILE"
        fi

    elif [[ "$EXPOSURE_CHOICE" == "2" ]]; then
        if [[ -z "$TS_HOST" ]]; then
            echo "Warning: tailscale connected but hostname not available — TLS cert setup skipped."
        else
            CERT_DIR="/etc/tailscale/certs"
            CERT_FILE="$CERT_DIR/${TS_HOST}.crt"
            KEY_FILE="$CERT_DIR/${TS_HOST}.key"
            CURRENT_USER=$(id -un)
            TIMER_UNIT="tailscale-cert-renewal"

            if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
                echo "TLS cert already present: $CERT_FILE"
                CERT_CONFIGURED=true
            else
                printf "Provision Tailscale TLS cert for %s? [Y/n]: " "$TS_HOST"
                read -r CERT_CHOICE
                if [[ "${CERT_CHOICE,,}" != "n" ]]; then
                    sudo mkdir -p "$CERT_DIR"
                    if sudo tailscale cert --cert-file "$CERT_FILE" --key-file "$KEY_FILE" "$TS_HOST"; then
                        sudo chmod 644 "$CERT_FILE"
                        sudo chmod 640 "$KEY_FILE"
                        sudo chown "root:${CURRENT_USER}" "$KEY_FILE"
                        echo "Cert provisioned: $CERT_FILE"
                        CERT_CONFIGURED=true
                    else
                        echo "Warning: cert provisioning failed — configure TLS manually." >&2
                        echo "  sudo tailscale cert --cert-file $CERT_FILE --key-file $KEY_FILE $TS_HOST" >&2
                    fi
                else
                    echo "Skipped. To provision later:"
                    echo "  sudo tailscale cert --cert-file $CERT_FILE --key-file $KEY_FILE $TS_HOST"
                fi
            fi

            if [[ "$CERT_CONFIGURED" == true ]]; then
                set_config_value "tls_cert" "$CERT_FILE"
                set_config_value "tls_key" "$KEY_FILE"
                set_config_value "bind_addr" "0.0.0.0"
                echo "Config updated: tls_cert, tls_key, bind_addr=0.0.0.0"
                echo "Note: this exposes ${SERVICE_PORT} to the whole LAN, not just the tailnet."

                # Install systemd renewal timer
                if systemctl is-active --quiet "${TIMER_UNIT}.timer" 2>/dev/null; then
                    echo "Renewal timer already active: ${TIMER_UNIT}.timer"
                else
                    printf "Install systemd renewal timer (weekly cert check)? [Y/n]: "
                    read -r TIMER_CHOICE
                    if [[ "${TIMER_CHOICE,,}" != "n" ]]; then
                        sudo tee /etc/systemd/system/${TIMER_UNIT}.service > /dev/null <<EOF
[Unit]
Description=Renew Tailscale TLS cert for voicetserver
After=network.target tailscaled.service

[Service]
Type=oneshot
ExecStart=/usr/bin/tailscale cert --cert-file ${CERT_FILE} --key-file ${KEY_FILE} ${TS_HOST}
ExecStartPost=/bin/chmod 640 ${KEY_FILE}
ExecStartPost=/bin/chown root:${CURRENT_USER} ${KEY_FILE}
EOF
                        sudo tee /etc/systemd/system/${TIMER_UNIT}.timer > /dev/null <<EOF
[Unit]
Description=Weekly Tailscale cert renewal for voicetserver

[Timer]
OnCalendar=weekly
Persistent=true

[Install]
WantedBy=timers.target
EOF
                        sudo systemctl daemon-reload
                        sudo systemctl enable --now "${TIMER_UNIT}.timer"
                        echo "Renewal timer enabled: ${TIMER_UNIT}.timer"
                    else
                        echo "Skipped. To install later: see CLAUDE.md"
                    fi
                fi
            fi
        fi
    else
        echo "Exposure setup skipped."
    fi
else
    echo "tailscale not found — exposure setup skipped."
    echo "  Install Tailscale and re-run the installer."
fi

# --- systemd unit for the server itself ---
#
# Without this the server only ever runs as long as the shell that started it, and after a
# reboot it is simply gone. The two Environment= lines are the crux: a login shell picks up
# LD_LIBRARY_PATH (CUDA) from ~/.bashrc or /etc/profile.d, a systemd service inherits
# nothing. Missing it, the server dies at startup with
#     Error: CublasError(CUBLAS_STATUS_NOT_INITIALIZED)
# which reads like a driver or GPU fault and is not one. We copy the values out of the shell
# running this installer, because that is an environment the binary is known to work in.
echo ""
UNIT_FILE="/etc/systemd/system/voicetserver.service"
BIN_PATH="$HOME/.local/bin/voicetserver"
if [[ -f "$UNIT_FILE" ]]; then
    echo "systemd unit already present: $UNIT_FILE"
elif [[ ! -x "$BIN_PATH" ]]; then
    echo "No binary at $BIN_PATH — skipping systemd unit."
else
    printf "Install systemd unit so the server starts at boot? [Y/n]: "
    read -r SVC_UNIT_CHOICE
    if [[ "${SVC_UNIT_CHOICE,,}" != "n" ]]; then
        {
            echo "[Unit]"
            echo "Description=voicetserver — two-engine ASR server"
            echo "After=network.target"
            echo "Wants=network.target"
            echo ""
            echo "[Service]"
            echo "Type=simple"
            echo "User=$(id -un)"
            echo "WorkingDirectory=$HOME"
            echo "ExecStart=${BIN_PATH}"
            [[ -n "${LD_LIBRARY_PATH:-}" ]] && echo "Environment=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
            echo "Environment=PATH=${PATH}"
            echo "Restart=always"
            echo "RestartSec=5"
            echo "NoNewPrivileges=true"
            echo ""
            echo "[Install]"
            echo "WantedBy=multi-user.target"
        } | sudo tee "$UNIT_FILE" > /dev/null

        if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
            echo "Note: LD_LIBRARY_PATH is empty in this shell, so the unit has no CUDA path."
            echo "  If the service fails with CUBLAS_STATUS_NOT_INITIALIZED, add it by hand."
        fi

        sudo systemctl daemon-reload
        echo "Unit written: $UNIT_FILE"
        echo ""
        echo "Only ONE instance may hold the GPU. Stop any hand-started server first —"
        echo "a second one fails with the same CUBLAS error, which looks like a unit fault"
        echo "but is just an occupied card (check with nvidia-smi):"
        echo "  pkill -x voicetserver && sleep 3"
        echo "  sudo systemctl enable --now voicetserver"
        echo "  journalctl -u voicetserver -f"
    else
        echo "Skipped. The server then only runs for as long as you keep a shell open."
    fi
fi

echo ""
echo "Done."
[[ -f "$HOME/.local/bin/voicetserver" ]] && echo "  Binary:   ~/.local/bin/voicetserver"
echo "  Venv:     $VENV_PATH"
echo "  Scripts:  ~/.config/voicetserver/tools/train_lora_{voxtral,qwen}.py"
echo "  Config:   $CONFIG_FILE"
[[ -n "$DATA_DIR"  ]] && echo "  Data:     $DATA_DIR"
[[ -n "$MODEL_DIR" ]] && echo "  Voxtral:  $MODEL_DIR"
if [[ -n "$QWEN_DIR" ]]; then
    echo "  Qwen3:    $QWEN_DIR"
else
    echo "  Qwen3:    disabled (qwen_model_dir not set)"
fi
if [[ "$CERT_CONFIGURED" == true ]]; then
    echo "  TLS cert: $CERT_FILE"
    echo "  External: wss://${TS_HOST}:8765"
fi
