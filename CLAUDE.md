# Build
cargo build --release --target-dir target

# Test
## Streaming (mic)
./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime
## Offline (WAV file)
./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime church.wav
## Full featured
./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime --hotkey F9 --type

# Model files
Dev: model weights are in Voxtral-Mini-4B-Realtime/ subfolder
Release: users place them next to voicet.exe (--model-dir defaults to ".")

# Release process
1. Build, then copy exe + CUDA DLLs + mel_filters.bin to release/ folder
2. Zip with PowerShell: Compress-Archive
3. gh release create vX.Y.Z <zip> --title "..." --notes "..."
4. Version in commit message, tag on commit, release tied to tag

# Folder structure notes
- target/release/ — cargo build output (has exe + model files for dev testing)
- release/ — distribution staging folder (exe + CUDA DLLs + mel_filters.bin)
- candle-fork/ — vendored candle dependency, don't update without reason
- Voxtral-Mini-4B-Realtime/ — gitignored, contains model weights for dev

# Docs
- ARCHITECTURE.md — current state reference (model, protocols, optimizations)
- PLAN_voicet-rust-rewrite.md — historical record of what was built per phase
- README.md — user-facing project overview

# GitHub
- Repo: github.com/Liddo-kun/voicet (public)
- Versioning: v0.3, v0.5, v0.5.1, v0.5.2 — include version in commit messages
