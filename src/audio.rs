// Audio decode for WebSocket audio frames.
//
// Phase 1: the browser AudioWorklet sends raw 32-bit float LE samples at 16kHz
// (mono, interleaved). No codec needed — bandwidth on Tailscale LAN is negligible
// at 16kHz * 4 bytes = 64 KB/s.
//
// Phase 3: replace with Opus decode (audiopus + WebM container parsing) to reduce
// bandwidth and match the browser MediaRecorder API.

/// Decode raw f32 LE PCM from a WebSocket binary frame (16kHz mono).
/// Each 4-byte chunk is one sample in IEEE 754 little-endian format.
pub fn decode_pcm_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
