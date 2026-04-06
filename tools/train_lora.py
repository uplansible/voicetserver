#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning for the Voxtral Mini 4B Realtime decoder.

Installation (run once):
    pip install uv                           # fast package installer
    uv pip install -r tools/requirements.txt
  or without uv:
    python3 -m venv tools/.venv && source tools/.venv/bin/activate
    pip install -r tools/requirements.txt

Usage (called by the server on POST /training/run, or manually):
    python3 tools/train_lora.py \
        --data-dir  ~/.config/voicetserver/training \
        --model-dir /path/to/Voxtral-Mini-4B-Realtime \
        --output-dir ~/.config/voicetserver/lora_adapter

Output:
    {output_dir}/adapter_model.safetensors   — LoRA weight pairs
    {output_dir}/adapter_config.json          — r, lora_alpha

The Rust server loads these at startup when lora_adapter is set in config.
Weight key naming: layers.{i}.attention.{wq,wk,wv,wo}.lora_{a,b}.weight
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# ---- Voxtral decoder constants (must match src/decoder.rs) ----
HIDDEN_SIZE     = 3072
INTERMEDIATE    = 9216
NUM_LAYERS      = 26
NUM_HEADS       = 32
NUM_KV_HEADS    = 8
HEAD_DIM        = 128
VOCAB_SIZE      = 131072
N_MELS          = 128
MEL_HOP         = 160
MEL_WIN         = 400
SAMPLE_RATE     = 16000

# ---- Audio helpers ----

def load_wav_f32(path: str) -> np.ndarray:
    """Load 16kHz mono WAV (16-bit int) as float32 numpy array."""
    import wave
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples[::n_channels]

    if framerate != SAMPLE_RATE:
        # Simple nearest-neighbour resampling
        ratio = SAMPLE_RATE / framerate
        new_len = int(len(samples) * ratio)
        indices = np.floor(np.arange(new_len) / ratio).astype(int).clip(0, len(samples) - 1)
        samples = samples[indices]

    return samples


def load_mel_filters(model_dir: str) -> np.ndarray:
    """Load precomputed Slaney mel filterbank from mel_filters.bin (128×201 f32 LE)."""
    path = Path(model_dir) / "mel_filters.bin"
    data = np.frombuffer(path.read_bytes(), dtype=np.float32)
    # 128 mel bands × 201 fft bins
    return data.reshape(128, 201)


def audio_to_mel(samples: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching the Rust mel.rs implementation."""
    n_fft  = 400
    hop    = 160
    win_fn = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    # STFT
    n_frames = (len(samples) - n_fft) // hop + 1
    frames   = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    ).copy()
    windowed = frames * win_fn
    fft_out  = np.fft.rfft(windowed, n=n_fft, axis=-1)
    power    = (np.abs(fft_out) ** 2).astype(np.float32)  # [n_frames, 201]

    mel      = filters @ power.T        # [128, n_frames]
    log_mel  = np.log(np.maximum(mel, 1e-10))
    return log_mel  # [128, n_frames]


# ---- Tokenizer ----

def load_tokenizer(model_dir: str):
    """Load SentencePiece tokenizer from tekken.json (Voxtral format)."""
    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("sentencepiece not installed — run: pip install sentencepiece")

    # tekken.json contains the SentencePiece model as base64
    tekken_path = Path(model_dir) / "tekken.json"
    with open(tekken_path) as f:
        tekken = json.load(f)

    import base64, tempfile
    sp_model_b64 = tekken.get("sp_model") or tekken.get("model")
    if sp_model_b64 is None:
        raise ValueError("tekken.json does not contain 'sp_model' or 'model' key")

    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as tmp:
        tmp.write(base64.b64decode(sp_model_b64))
        tmp_path = tmp.name

    sp = spm.SentencePieceProcessor()
    sp.Load(tmp_path)
    os.unlink(tmp_path)
    return sp


def tokenize(sp, text: str) -> list[int]:
    """Tokenize text with BOS prepended (matching Voxtral streaming protocol)."""
    ids = sp.EncodeAsIds(text)
    return [1] + ids + [2]  # BOS=1, EOS=2


# ---- Deinterleave helper (must match Rust deinterleave_qk) ----

def deinterleave_qk(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder GQA interleaved weight matrix to match Rust decoder_qk deinterleaving.

    Voxtral stores Q/K heads interleaved as: h0_1, h0_2, h1_1, h1_2, ...
    The Rust code reorders them to canonical head order.
    This function applies the same reordering to stay consistent.
    """
    out_features, in_features = weight.shape
    # Split each head's two sub-halves and recombine
    w = weight.reshape(n_heads, 2, head_dim // 2, in_features)
    w = w.permute(1, 0, 2, 3).reshape(out_features, in_features)
    return w.contiguous()


# ---- Minimal decoder forward (for computing CE loss) ----

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class LoRALinear(nn.Module):
    """Linear layer with LoRA delta. base_weight is frozen; lora_a/lora_b are trained."""
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        out_f, in_f = weight.shape
        self.base   = nn.Parameter(weight, requires_grad=False)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank))
        self.scale  = alpha / rank
        # Kaiming uniform init for lora_a (standard PEFT init)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class Attention(nn.Module):
    def __init__(self, weights: dict, layer_idx: int, rank: int, alpha: float,
                 target_modules: list[str]):
        super().__init__()
        prefix = f"layers.{layer_idx}.attention"
        for proj in ('wq', 'wk', 'wv', 'wo'):
            w = weights[f"{prefix}.{proj}.weight"]
            if proj in ('wq', 'wk'):
                n_h = NUM_HEADS if proj == 'wq' else NUM_KV_HEADS
                w   = deinterleave_qk(w, n_h, HEAD_DIM)
            if proj in target_modules:
                setattr(self, proj, LoRALinear(w, rank, alpha))
            else:
                # Frozen plain linear
                lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
                lin.weight = nn.Parameter(w, requires_grad=False)
                setattr(self, proj, lin)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, NUM_HEADS,    HEAD_DIM)
        k = self.wk(x).reshape(B, T, NUM_KV_HEADS, HEAD_DIM)
        v = self.wv(x).reshape(B, T, NUM_KV_HEADS, HEAD_DIM)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads for GQA
        rep = NUM_HEADS // NUM_KV_HEADS
        k   = k.repeat_interleave(rep, dim=2)
        v   = v.repeat_interleave(rep, dim=2)

        # [B, heads, T, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = HEAD_DIM ** -0.5
        attn  = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
        out   = attn.permute(0, 2, 1, 3).reshape(B, T, NUM_HEADS * HEAD_DIM)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, weights: dict, layer_idx: int):
        super().__init__()
        prefix = f"layers.{layer_idx}.feed_forward"
        for name in ('w1', 'w2', 'w3'):
            w   = weights[f"{prefix}.{name}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight = nn.Parameter(w, requires_grad=False)
            setattr(self, name, lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    def __init__(self, weights: dict, layer_idx: int, rank: int, alpha: float,
                 target_modules: list[str]):
        super().__init__()
        i = layer_idx
        self.attn      = Attention(weights, i, rank, alpha, target_modules)
        self.mlp       = MLP(weights, i)
        self.attn_norm = RMSNorm(HIDDEN_SIZE)
        self.ffn_norm  = RMSNorm(HIDDEN_SIZE)
        self.attn_norm.weight = nn.Parameter(weights[f"layers.{i}.attention_norm.weight"], requires_grad=False)
        self.ffn_norm.weight  = nn.Parameter(weights[f"layers.{i}.ffn_norm.weight"],      requires_grad=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), positions, cos, sin)
        return h + self.mlp(self.ffn_norm(h))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding. x: [B, T, heads, head_dim]."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    c  = cos[:, :x.shape[1], :, :d // 2]
    s  = sin[:, :x.shape[1], :, :d // 2]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def build_rope(max_seq: int, device: torch.device, dtype: torch.dtype,
               theta: float = 1_000_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cos/sin caches. Returns (cos, sin) each [1, max_seq, 1, head_dim//2]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    t        = torch.arange(max_seq, dtype=torch.float32)
    freqs    = torch.outer(t, inv_freq)  # [max_seq, head_dim//2]
    cos      = freqs.cos().to(dtype).to(device).unsqueeze(0).unsqueeze(2)  # [1, T, 1, d/2]
    sin      = freqs.sin().to(dtype).to(device).unsqueeze(0).unsqueeze(2)
    return cos, sin


# ---- Training ----

def load_training_pairs(data_dir: str) -> list[dict]:
    pairs_path = Path(data_dir) / "pairs.jsonl"
    if not pairs_path.exists():
        sys.exit(f"No training data found at {pairs_path}")
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load model weights
    print("Loading model weights...")
    st_path = Path(args.model_dir) / "consolidated.safetensors"
    if not st_path.exists():
        sys.exit(f"Model not found: {st_path}")
    weights = load_file(str(st_path), device=str(device))
    weights = {k: v.to(dtype) for k, v in weights.items()}

    # Load tokenizer and mel filters
    sp      = load_tokenizer(args.model_dir)
    filters = load_mel_filters(args.model_dir)

    # Load training pairs
    pairs = load_training_pairs(args.data_dir)
    print(f"Training pairs: {len(pairs)}")
    if len(pairs) == 0:
        sys.exit("No training pairs found.")

    # Embedding matrix (tied weights)
    tok_emb_key = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
    tok_embeddings = nn.Parameter(weights[tok_emb_key], requires_grad=False)

    # Build decoder layers with LoRA on target modules
    target_modules = args.target_modules.split(",")
    print(f"LoRA rank={args.rank}, alpha={args.lora_alpha}, targets={target_modules}")
    layers   = nn.ModuleList([
        DecoderLayer(weights, i, args.rank, args.lora_alpha, target_modules)
        for i in tqdm(range(NUM_LAYERS), desc="Loading decoder layers")
    ])
    final_norm = RMSNorm(HIDDEN_SIZE)
    final_norm.weight = nn.Parameter(weights["norm.weight"], requires_grad=False)

    layers.to(device)
    final_norm.to(device)
    tok_embeddings = tok_embeddings.to(device)

    # Collect only LoRA parameters for optimizer
    lora_params = [p for p in layers.parameters() if p.requires_grad]
    print(f"LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    if not lora_params:
        sys.exit("No LoRA parameters found — check target_modules")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    cos_cache, sin_cache = build_rope(2048, device, dtype)
    positions = torch.arange(2048, device=device).unsqueeze(0)

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0.0
        for pair in tqdm(pairs, desc=f"Epoch {epoch+1}/{args.epochs}"):
            audio_path = Path(args.data_dir) / "audio" / f"{pair['id']}.wav"
            if not audio_path.exists():
                print(f"Warning: audio file missing: {audio_path}", file=sys.stderr)
                continue

            samples  = load_wav_f32(str(audio_path))
            log_mel  = audio_to_mel(samples, filters)  # [128, T_mel]
            token_ids = tokenize(sp, pair["text"])

            if len(token_ids) < 2:
                continue

            # Input tokens (all but last), targets (all but first)
            input_ids  = torch.tensor(token_ids[:-1], device=device).unsqueeze(0)  # [1, T-1]
            target_ids = torch.tensor(token_ids[1:],  device=device)                # [T-1]

            # Token embeddings
            x = tok_embeddings[input_ids.flatten()].unsqueeze(0).to(dtype)  # [1, T-1, H]
            T = x.shape[1]

            # Forward through decoder
            for layer in layers:
                x = layer(x, positions[:, :T], cos_cache[:, :T], sin_cache[:, :T])
            x = final_norm(x)  # [1, T-1, H]

            # LM head (tied weights)
            logits = x @ tok_embeddings.T  # [1, T-1, V]
            logits = logits.squeeze(0)      # [T-1, V]

            loss = F.cross_entropy(logits.float(), target_ids)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(pairs), 1)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    # Save adapter
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_tensors = {}
    for i, layer in enumerate(layers):
        for proj_name in ('wq', 'wk', 'wv', 'wo'):
            proj = getattr(layer.attn, proj_name, None)
            if isinstance(proj, LoRALinear):
                adapter_tensors[f"layers.{i}.attention.{proj_name}.lora_a.weight"] = proj.lora_a.data.contiguous().bfloat16()
                adapter_tensors[f"layers.{i}.attention.{proj_name}.lora_b.weight"] = proj.lora_b.data.contiguous().bfloat16()

    save_file(adapter_tensors, str(output_dir / "adapter_model.safetensors"))

    cfg = {"r": args.rank, "lora_alpha": args.lora_alpha, "target_modules": target_modules}
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nAdapter saved to {output_dir}")
    print(f"  {len(adapter_tensors)} tensors, {sum(t.numel() for t in adapter_tensors.values()):,} parameters")
    print("Reload the server (or set lora_adapter in config) to apply.")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Voxtral Mini 4B decoder")
    parser.add_argument("--data-dir",       required=True, help="Path to training data directory")
    parser.add_argument("--model-dir",      required=True, help="Path to Voxtral model directory")
    parser.add_argument("--output-dir",     required=True, help="Output path for LoRA adapter")
    parser.add_argument("--rank",           type=int,   default=8,     help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha",     type=float, default=16.0,  help="LoRA alpha (default: 16)")
    parser.add_argument("--lr",             type=float, default=1e-4,  help="Learning rate (default: 1e-4)")
    parser.add_argument("--epochs",         type=int,   default=3,     help="Training epochs (default: 3)")
    parser.add_argument("--target-modules", default="wq,wv",
                        help="Comma-separated attention projections to adapt (default: wq,wv)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
