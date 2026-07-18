// Tekken tokenizer: decode-only for Voxtral Realtime
// Token IDs 0-999: special tokens (rank = token ID)
// Token IDs 1000-131071: BPE tokens (ID - 1000 = rank in vocab array)

use anyhow::{Context, Result};
use base64::Engine;

pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
#[allow(dead_code)]
pub const AUDIO_ID: u32 = 24;        // audio frame token (encoder output marker)
#[allow(dead_code)]
pub const BEGIN_AUDIO_ID: u32 = 25;  // begin-audio boundary token
pub const STREAMING_PAD_ID: u32 = 32;
pub const STREAMING_WORD_ID: u32 = 33;

const NUM_SPECIAL: u32 = 1000;

pub struct Tokenizer {
    /// BPE vocab: index = rank, value = token bytes
    vocab: Vec<Vec<u8>>,
}

/// Minimal structure to deserialize only what we need from tekken.json,
/// avoiding a full serde_json::Value DOM (~50-75MB for 15MB file).
#[derive(serde::Deserialize)]
struct TekkenFile {
    vocab: Vec<VocabEntry>,
}

#[derive(serde::Deserialize)]
struct VocabEntry {
    token_bytes: String,
}

impl Tokenizer {
    pub fn load(model_dir: &str) -> Result<Self> {
        let path = format!("{model_dir}/tekken.json");
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {path}"))?;
        let parsed: TekkenFile = serde_json::from_str(&data)
            .context("parsing tekken.json")?;
        drop(data);

        let engine = base64::engine::general_purpose::STANDARD;
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(parsed.vocab.len());

        for entry in &parsed.vocab {
            let bytes = engine.decode(&entry.token_bytes)
                .context("base64 decode failed")?;
            vocab.push(bytes);
        }
        drop(parsed);

        Ok(Self { vocab })
    }

    /// Decode a single token ID to bytes. Returns None for special tokens that should be skipped.
    pub fn decode_token(&self, id: u32) -> Option<Vec<u8>> {
        if id == STREAMING_PAD_ID {
            return None; // skip padding tokens
        }
        if id == STREAMING_WORD_ID {
            return None; // control token, not text — BPE tokens carry their own spacing
        }
        if id < NUM_SPECIAL {
            return None; // other special tokens: skip
        }
        let rank = (id - NUM_SPECIAL) as usize;
        if rank < self.vocab.len() {
            Some(self.vocab[rank].clone())
        } else {
            None
        }
    }

    /// Encode text via greedy longest-match against the BPE vocab.
    ///
    /// Not true BPE (no merge ranks), but every returned ID is a valid vocab token
    /// and the decoded concatenation equals the input text. Sufficient for prefill
    /// priming, where the tokens only need to *read* as the intended text — the
    /// exact segmentation the trained model would produce is not required.
    pub fn encode_greedy(&self, text: &str) -> Vec<u32> {
        use std::collections::HashMap;
        let map: HashMap<&[u8], u32> = self.vocab.iter().enumerate()
            .map(|(rank, bytes)| (bytes.as_slice(), rank as u32 + NUM_SPECIAL))
            .collect();
        let max_len = self.vocab.iter().map(|b| b.len()).max().unwrap_or(1);
        let bytes = text.as_bytes();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let end = (i + max_len).min(bytes.len());
            let matched = (i + 1..=end).rev()
                .find_map(|j| map.get(&bytes[i..j]).map(|&id| (id, j)));
            match matched {
                Some((id, j)) => { ids.push(id); i = j; }
                // Byte not in vocab — skip it (cannot happen for a byte-level BPE
                // vocab that covers all single bytes, but avoids an infinite loop).
                None => { i += 1; }
            }
        }
        ids
    }

    /// Decode a sequence of token IDs to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.decode_token(id) {
                bytes.extend_from_slice(&b);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}
