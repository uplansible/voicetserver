// Audio decode for voicetserver.
//
// Two entry points:
//
//   decode_pcm_f32(data)    — raw f32 LE PCM (16kHz mono) from ASR WebSocket.
//                             Browser AudioWorklet sends 64 KB/s on Tailscale LAN.
//
//   decode_audio_bytes(data) — multi-codec decode via symphonia for training pair uploads.
//                              Accepts OGG Vorbis (Firefox MediaRecorder), WAV, and other
//                              formats symphonia supports. Falls back to raw PCM.
//                              Output is always resampled to 16kHz mono f32.

/// Decode raw f32 LE PCM from a WebSocket binary frame (16kHz mono).
/// Each 4-byte chunk is one sample in IEEE 754 little-endian format.
pub fn decode_pcm_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Decode audio from a byte buffer (OGG Vorbis, WAV, WebM, or raw f32 PCM fallback).
/// Returns 16kHz mono f32 samples. Used for training pair uploads.
pub fn decode_audio_bytes(data: &[u8]) -> Vec<f32> {
    match decode_with_symphonia(data) {
        Ok(samples) => samples,
        Err(_)      => decode_pcm_f32(data),  // raw f32 LE PCM fallback
    }
}

fn decode_with_symphonia(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    use symphonia::core::{
        audio::SampleBuffer,
        codecs::DecoderOptions,
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    };

    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| anyhow::anyhow!("symphonia probe: {e}"))?;

    let mut format  = probed.format;
    let track       = format.default_track()
        .ok_or_else(|| anyhow::anyhow!("no audio track found"))?;
    let track_id    = track.id;
    let src_rate    = track.codec_params.sample_rate.unwrap_or(16000) as usize;
    let n_channels  = track.codec_params.channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow::anyhow!("symphonia decoder: {e}"))?;

    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p)  => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }

        let decoded = match decoder.decode(&packet) {
            Ok(d)  => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(buf.samples());
    }

    if interleaved.is_empty() {
        anyhow::bail!("no audio decoded");
    }

    // Downmix to mono
    let mono: Vec<f32> = if n_channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(n_channels)
            .map(|ch| ch.iter().sum::<f32>() / n_channels as f32)
            .collect()
    };

    // Resample to 16kHz (nearest-neighbour — sufficient for voice)
    if src_rate == 16000 {
        return Ok(mono);
    }
    let ratio    = src_rate as f64 / 16000.0;
    let new_len  = (mono.len() as f64 / ratio) as usize;
    let resampled: Vec<f32> = (0..new_len)
        .map(|i| {
            let src = ((i as f64 * ratio) as usize).min(mono.len() - 1);
            mono[src]
        })
        .collect();
    Ok(resampled)
}
