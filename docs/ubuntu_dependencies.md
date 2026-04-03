# Ubuntu Server Dependencies — SCHMIDIspeech

This file lists every Ubuntu package required to build and run voicetserver, grouped
by phase and crate. Install on the production GPU server before building.

## Phase 0 — Build toolchain

| Package | Crate / purpose |
|---------|-----------------|
| `pkg-config` | Required by many crates to locate system libraries |
| `build-essential` | gcc, make (required by cudaforge for CUDA kernel compilation) |

```bash
apt install -y pkg-config build-essential
```

## Phase 0 — CUDA (production GPU server only; skip in dev Docker)

| Package | Crate / purpose |
|---------|-----------------|
| `nvidia-cuda-toolkit` | `candle-core/cuda`, `candle-flash-attn`, `cudaforge` (nvcc) |

The CUDA toolkit must provide `nvcc` in `PATH` **and** the runtime libraries. Install
from NVIDIA's official apt repository (not from the Ubuntu universe repo — version is
too old):

```bash
# Add NVIDIA CUDA repo (example for Ubuntu 22.04 + CUDA 12.x):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-x   # pick the version matching your driver

# Then add to ~/.bashrc or /etc/environment:
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

Build with CUDA:
```bash
export CUDA_PATH=/usr/local/cuda
cargo build --release --features cuda
```

## Phase 1 — WebSocket server + TLS

| Package | Crate / purpose |
|---------|-----------------|
| `libssl-dev` | `rustls` / `axum-server` TLS backend |

```bash
apt install -y libssl-dev
```

## Removed dependencies (stripped in Phase 0)

These were required by the upstream voicet desktop app and are **no longer needed**:

| Package | Was required by |
|---------|----------------|
| `libasound2-dev` | `cpal` (microphone capture) |
| `libx11-dev` | `rdev`, `enigo` (X11 hotkey/input) |
| `libxtst-dev` | `rdev` (X11 Xtst extension) |
| `libxdo-dev` | `enigo` (libxdo keystroke injection) |

## Full install command (production GPU server)

```bash
apt install -y pkg-config build-essential libssl-dev
# Plus CUDA toolkit from NVIDIA repo (see above)
```

## Dev Docker (no GPU)

```bash
apt install -y pkg-config build-essential libssl-dev
# No CUDA needed — build with:  cargo build  (no --features cuda)
```
