#!/usr/bin/env bash
# Installs voicetserver binary, LoRA venv, train_lora.py, and model files; writes config.toml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$HOME/.config/voicetserver/config.toml"
DEFAULT_VENV="$HOME/.local/share/voicetserver-venv"
DEFAULT_MODEL_DIR="$HOME/models/Voxtral-Mini-4B-Realtime"
HF_BASE="https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/main"

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
# Restart required for: model_dir, device, port, bind_addr, tls_cert, tls_key, lora_adapter, venv_path
# Runtime-adjustable via PATCH /config: delay, silence_threshold, silence_flush, min_speech, rms_ema

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

# --- Model files ---
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

# --- TLS certificate via tailscale cert ---
echo ""
CERT_CONFIGURED=false
if command -v tailscale &>/dev/null; then
    TS_HOST=$(tailscale status --json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" \
        2>/dev/null || true)

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
    echo "tailscale not found — TLS cert setup skipped."
    echo "  Install Tailscale and re-run the installer."
fi

echo ""
echo "Done."
[[ -f "$HOME/.local/bin/voicetserver" ]] && echo "  Binary:   ~/.local/bin/voicetserver"
echo "  Venv:     $VENV_PATH"
echo "  Scripts:  ~/.config/voicetserver/tools/train_lora_{voxtral,qwen}.py"
echo "  Config:   $CONFIG_FILE"
[[ -n "$DATA_DIR"  ]] && echo "  Data:     $DATA_DIR"
[[ -n "$MODEL_DIR" ]] && echo "  Models:   $MODEL_DIR"
if [[ "$CERT_CONFIGURED" == true ]]; then
    echo "  TLS cert: $CERT_FILE"
    echo "  External: wss://${TS_HOST}:8765"
fi
