# SCHMIDIspeech — Task List

## Phase 0 — Strip desktop layer
- [x] 0.1 Read ~/voicet fully (CLAUDE.md, ARCHITECTURE.md, all src/ files)
- [x] 0.2 Read current voicetserver state (Cargo.toml, src/, README.md)
- [x] 0.3 Rename binary to voicetserver; add cuda/cpu feature flags in Cargo.toml
- [x] 0.4 Remove GUI/input deps (tray-icon, eframe, winit, rdev, enigo, cpal) from Cargo.toml; add server deps
- [x] 0.5 Guard build.rs CUDA kernel compilation behind cuda feature
- [x] 0.6 Delete tray.rs, hotkey.rs, settings_window.rs
- [x] 0.7 Strip settings.rs of hotkey/type_mode/GUI fields
- [x] 0.8 Strip main.rs of GUI/hotkey/type dispatch; add async tokio entry
- [x] 0.9 Gut streaming.rs: remove cpal/hotkey/OutputSink; export StreamingState
- [x] 0.10 Wrap m1_attention and flash-attn call sites in cuda feature cfg; add CPU fallback
- [x] 0.11 cargo check passes in Docker (no cuda feature)
- [x] 0.12 Create docs/ubuntu_dependencies.md

## Phase 1 — WebSocket server with TLS
- [x] 1.1 Create src/audio.rs (raw f32 PCM decode; Opus planned Phase 3)
- [x] 1.2 Implement StreamingState::new_sync + process_chunk_sync in streaming.rs
- [x] 1.3 Create server module in main.rs (axum WS + TLS, GPU mutex via model.inner lock, health endpoint)
- [x] 1.4 Wire server into main.rs (tokio::main, Arc<VoxtralModel>, server::run)
- [x] 1.5 Add new CLI flags (--port, --bind-addr, --tls-cert, --tls-key, --lora-adapter)
- [x] 1.6 Create config/medical_terms_de.txt (80+ German medical terms)
- [x] 1.7 Create schmidispeech.user.js (Violentmonkey userscript)
- [x] 1.8 Rewrite README.md
- [x] 1.9 Create stub src/session.rs (Phase 2 placeholder)
- [x] 1.10 Create stub src/macros.rs (Phase 4 placeholder)
- [ ] 1.11 cargo build --features cuda on GPU server + end-to-end browser test

## Phase 2 — Patient session vocabulary (stub)
- [ ] 2.1 Create docs/phase2_session_vocab.md
- [ ] 2.2 Implement session.rs data structures and route handlers (feature-flagged)

## Phase 3 — Voice calibration, correction layer, LoRA
- [ ] 3.1 Create tools/calibrate_silence.py
- [ ] 3.2 Implement src/correction.rs (Aho-Corasick, corrections.toml, SIGHUP reload)
- [ ] 3.3 Wire config/medical_terms_de.txt into correction automaton
- [ ] 3.4 Create tools/generate_corrections.py (Anthropic API)
- [ ] 3.5 Implement src/lora.rs (weight merge for fine-tuned adapters)
- [ ] 3.6 Create tools/generate_training_sentences.py
- [ ] 3.7 Create tools/train_lora.py
- [ ] 3.8 Replace raw PCM with Opus/WebM decode in audio.rs (audiopus + matroska)

## Phase 4 — Post-processing and macros
- [ ] 4.1 Create docs/phase4_macros.md
- [ ] 4.2 Implement src/macros.rs (config/macros.json expansion)

## Phase 5 — Nice-to-haves (docs only)
- [ ] 5.1 Create docs/phase5_ideas.md
