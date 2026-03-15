#!/usr/bin/env bash
# build-linux.sh — Build and package voicet for Linux (x86_64 or aarch64 + CUDA)
#
# Usage:
#   ./scripts/build-linux.sh              # Build only
#   ./scripts/build-linux.sh --package    # Build + create release tarball
#
# Prerequisites:
#   - Rust toolchain (rustup.rs)
#   - CUDA Toolkit 12.x+ (nvcc in PATH)
#   - System packages:
#       Ubuntu/Debian: apt install pkg-config libasound2-dev libx11-dev libxtst-dev libxdo-dev
#       Fedora/RHEL:   dnf install alsa-lib-devel libX11-devel libXtst-devel libxdo-devel
#
# Environment:
#   CUDA_PATH  — Override auto-detected CUDA location (default: /usr/local/cuda)
#   VERSION    — Override version string (default: read from Cargo.toml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Detect architecture ---
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  TARGET_ARCH="x64" ;;
    aarch64) TARGET_ARCH="arm64" ;;
    *)       echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

echo "=== Voicet Linux Build ==="
echo "Architecture: $ARCH ($TARGET_ARCH)"

# --- Find CUDA ---
if [ -z "${CUDA_PATH:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_PATH="/usr/local/cuda"
    elif command -v nvcc &>/dev/null; then
        export CUDA_PATH="$(dirname "$(dirname "$(which nvcc)")")"
    else
        echo "ERROR: CUDA not found. Install CUDA Toolkit or set CUDA_PATH."
        exit 1
    fi
fi
echo "CUDA_PATH: $CUDA_PATH"

# Ensure nvcc is in PATH
if ! command -v nvcc &>/dev/null; then
    export PATH="$CUDA_PATH/bin:$PATH"
fi
echo "nvcc: $(nvcc --version 2>&1 | grep release)"

# --- Check build dependencies ---
echo ""
echo "Checking build dependencies..."
MISSING=()

# ALSA (cpal)
pkg-config --exists alsa 2>/dev/null || MISSING+=("libasound2-dev")
# X11 (rdev)
pkg-config --exists x11 2>/dev/null || MISSING+=("libx11-dev")
# Xtst (rdev)
pkg-config --exists xtst 2>/dev/null || MISSING+=("libxtst-dev")
# libxdo (enigo)
if ! pkg-config --exists libxdo 2>/dev/null && ! ldconfig -p 2>/dev/null | grep -q libxdo; then
    MISSING+=("libxdo-dev")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing: ${MISSING[*]}"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install ${MISSING[*]}"
    echo "  Fedora/RHEL:   sudo dnf install $(echo "${MISSING[*]}" | sed 's/-dev/-devel/g')"
    exit 1
fi
echo "All build dependencies found."

# --- Build ---
echo ""
echo "Building voicet (release, LTO)..."
cd "$PROJECT_DIR"
cargo build --release

BINARY="$PROJECT_DIR/target/release/voicet"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: No binary at $BINARY"
    exit 1
fi
echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"

# --- Package ---
if [[ "${1:-}" == "--package" ]]; then
    VERSION="${VERSION:-$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')}"
    RELEASE_NAME="voicet-v${VERSION}-linux-${TARGET_ARCH}-cuda"
    RELEASE_DIR="$PROJECT_DIR/release-linux"

    echo ""
    echo "=== Packaging: $RELEASE_NAME ==="

    rm -rf "$RELEASE_DIR"
    mkdir -p "$RELEASE_DIR"

    # Binary (stripped)
    cp "$BINARY" "$RELEASE_DIR/voicet"
    strip "$RELEASE_DIR/voicet"
    echo "voicet (stripped): $(du -h "$RELEASE_DIR/voicet" | cut -f1)"

    # mel_filters.bin
    if [ -f "$PROJECT_DIR/mel_filters.bin" ]; then
        cp "$PROJECT_DIR/mel_filters.bin" "$RELEASE_DIR/"
        echo "mel_filters.bin: $(du -h "$RELEASE_DIR/mel_filters.bin" | cut -f1)"
    else
        echo "WARNING: mel_filters.bin not found in project root"
    fi

    # Bundle CUDA shared libraries
    echo ""
    echo "Bundling CUDA libraries..."
    CUDA_LIB="$CUDA_PATH/lib64"
    [ -d "$CUDA_LIB" ] || CUDA_LIB="$CUDA_PATH/lib"

    for lib in libcublas.so libcublasLt.so libcudart.so libcurand.so; do
        # Find the real (versioned) .so, not symlinks
        FOUND=$(find "$CUDA_LIB" -name "${lib}.*" -not -type l 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            BASENAME=$(basename "$FOUND")
            cp "$FOUND" "$RELEASE_DIR/"
            # Create unversioned symlink so the dynamic linker finds it
            if [ "$BASENAME" != "$lib" ]; then
                ln -sf "$BASENAME" "$RELEASE_DIR/$lib"
            fi
            echo "  $BASENAME ($(du -h "$FOUND" | cut -f1))"
        else
            echo "  WARNING: $lib not found in $CUDA_LIB"
        fi
    done

    # Create wrapper script that sets LD_LIBRARY_PATH
    cat > "$RELEASE_DIR/run-voicet.sh" << 'WRAPPER'
#!/usr/bin/env bash
# Run voicet with bundled CUDA libraries
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DIR/voicet" "$@"
WRAPPER
    chmod +x "$RELEASE_DIR/run-voicet.sh"

    # Create tarball
    cd "$PROJECT_DIR"
    tar czf "${RELEASE_NAME}.tar.gz" -C release-linux .
    echo ""
    echo "Release: ${RELEASE_NAME}.tar.gz ($(du -h "${RELEASE_NAME}.tar.gz" | cut -f1))"
    echo ""
    echo "Contents:"
    ls -lh "$RELEASE_DIR/"
    echo ""
    echo "Users extract and run:"
    echo "  tar xzf ${RELEASE_NAME}.tar.gz"
    echo "  # Copy model files (consolidated.safetensors, tekken.json) into same directory"
    echo "  ./run-voicet.sh                    # streaming mode"
    echo "  ./run-voicet.sh recording.wav      # offline mode"
fi

echo ""
echo "=== Done ==="
