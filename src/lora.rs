// LoRA (Low-Rank Adaptation) adapter loading for the Voxtral decoder.
//
// Adapter directory layout (produced by tools/train_lora.py):
//   adapter_config.json          — {"r": 8, "lora_alpha": 16}
//   adapter_model.safetensors    — lora_a / lora_b weight pairs
//
// Weight key naming convention (matches train_lora.py output):
//   layers.{i}.attention.wq.lora_a.weight   [rank, in_features]
//   layers.{i}.attention.wq.lora_b.weight   [out_features, rank]
//   (same for wk, wv, wo)
//
// At inference time the delta is applied as:
//   proj_output += scale * lora_b @ lora_a @ input
//
// where scale = lora_alpha / r.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use serde::Deserialize;

#[derive(Deserialize)]
struct AdapterConfig {
    r: usize,
    lora_alpha: f32,
}

/// Pre-composed LoRA delta for a single linear layer.
/// Stored as combined = scale * lora_b @ lora_a, shape [out_features, in_features].
/// Applied as: `output = base.forward(x) + combined.forward(x)`.
pub struct LoraLinear(pub Linear);

/// LoRA adapters for all attention projections in one decoder layer.
pub struct AttentionLora {
    pub wq: Option<LoraLinear>,
    pub wk: Option<LoraLinear>,
    pub wv: Option<LoraLinear>,
    pub wo: Option<LoraLinear>,
}

/// LoRA adapters for the full decoder (one AttentionLora per layer).
pub struct DecoderLora {
    pub layers: Vec<AttentionLora>,
}

/// Load a LoRA adapter from `adapter_dir`.
///
/// Returns `None` if the directory does not exist (no adapter configured).
pub fn load_decoder_lora(adapter_dir: &Path, device: &Device, dtype: DType) -> Result<Option<DecoderLora>> {
    if !adapter_dir.exists() {
        return Ok(None);
    }

    // Load adapter_config.json
    let cfg_path = adapter_dir.join("adapter_config.json");
    if !cfg_path.exists() {
        anyhow::bail!("LoRA adapter_config.json not found in {}", adapter_dir.display());
    }
    let cfg: AdapterConfig = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)
        .map_err(|e| anyhow::anyhow!("adapter_config.json parse error: {e}"))?;
    let scale = cfg.lora_alpha / cfg.r as f32;

    // Load adapter_model.safetensors
    let st_path = adapter_dir.join("adapter_model.safetensors");
    if !st_path.exists() {
        anyhow::bail!("adapter_model.safetensors not found in {}", adapter_dir.display());
    }
    let data = std::fs::read(&st_path)?;
    let tensors = load_safetensors_tensors(&data, device, dtype)?;

    // Build per-layer LoRA structs
    let mut layers = Vec::new();
    for i in 0..crate::decoder::NUM_LAYERS {
        let attn = AttentionLora {
            wq: build_combined(&tensors, &format!("layers.{i}.attention.wq"), scale, device, dtype)?,
            wk: build_combined(&tensors, &format!("layers.{i}.attention.wk"), scale, device, dtype)?,
            wv: build_combined(&tensors, &format!("layers.{i}.attention.wv"), scale, device, dtype)?,
            wo: build_combined(&tensors, &format!("layers.{i}.attention.wo"), scale, device, dtype)?,
        };
        layers.push(attn);
    }

    eprintln!("LoRA adapter loaded: r={}, alpha={}, {} layers", cfg.r, cfg.lora_alpha, layers.len());
    Ok(Some(DecoderLora { layers }))
}

/// Load all tensors from a safetensors byte buffer, converting to `dtype` on `device`.
fn load_safetensors_tensors(
    data: &[u8],
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let st = safetensors::SafeTensors::deserialize(data)
        .map_err(|e| anyhow::anyhow!("safetensors deserialize error: {e}"))?;

    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        let candle_dtype = st_dtype_to_candle(view.dtype())?;
        let tensor = Tensor::from_raw_buffer(view.data(), candle_dtype, view.shape(), device)
            .map_err(|e| anyhow::anyhow!("tensor '{}' load error: {e}", name))?;
        let tensor = tensor.to_dtype(dtype)
            .map_err(|e| anyhow::anyhow!("tensor '{}' dtype conversion error: {e}", name))?;
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

/// Build the pre-composed LoRA delta: combined = scale * lora_b @ lora_a.
/// Returns None if neither lora_a nor lora_b exists for this prefix (layer has no LoRA).
fn build_combined(
    tensors: &HashMap<String, Tensor>,
    key_prefix: &str,
    scale: f32,
    _device: &Device,
    _dtype: DType,
) -> Result<Option<LoraLinear>> {
    let a_key = format!("{key_prefix}.lora_a.weight");
    let b_key = format!("{key_prefix}.lora_b.weight");

    match (tensors.get(&a_key), tensors.get(&b_key)) {
        (Some(a), Some(b)) => {
            // a: [rank, in_features], b: [out_features, rank]
            // combined = b @ a → [out_features, in_features]
            let combined = b.matmul(a)
                .map_err(|e| anyhow::anyhow!("LoRA matmul for '{}': {e}", key_prefix))?;
            let combined = (combined * scale as f64)
                .map_err(|e| anyhow::anyhow!("LoRA scale for '{}': {e}", key_prefix))?;
            Ok(Some(LoraLinear(Linear::new(combined, None))))
        }
        (None, None) => Ok(None),
        _ => anyhow::bail!(
            "LoRA: found only one of '{}' / '{}' — adapter may be corrupt",
            a_key, b_key
        ),
    }
}

fn st_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32  => Ok(DType::F32),
        safetensors::Dtype::F16  => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        other => anyhow::bail!("unsupported safetensors dtype for LoRA: {:?}", other),
    }
}
