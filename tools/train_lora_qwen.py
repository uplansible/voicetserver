#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning for Qwen3-ASR-0.6B text decoder.

Loads model.safetensors directly (no transformers model class needed).
Requires: torch, safetensors, tokenizers, numpy, tqdm

Output (compatible with schmidiscribe Rust runtime):
  adapter_model.safetensors — keys: layers.{i}.self_attn.{proj}.lora_{a,b}
  adapter_config.json       — {"r": N, "lora_alpha": F, "target_modules": [...]}

Architecture (Qwen3-ASR-0.6B config.json):
  Text decoder: 28 layers, hidden=1024, 16 Q-heads, 8 KV-heads, head_dim=128
  Audio encoder: 18 layers, d_model=896, 14 heads, output_dim=1024
"""

import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# ── Architecture constants ────────────────────────────────────────────────────

TEXT_HIDDEN   = 1024
TEXT_LAYERS   = 28
TEXT_Q_HEADS  = 16
TEXT_KV_HEADS = 8
TEXT_HEAD_DIM = 128
TEXT_VOCAB    = 151936
TEXT_RMS_EPS  = 1e-6
ROPE_THETA    = 1_000_000

AUDIO_D_MODEL = 896
AUDIO_HEADS   = 14
AUDIO_LAYERS  = 18
AUDIO_OUTPUT  = 1024   # == TEXT_HIDDEN
CONV_CHANNELS = 480

MEL_BINS   = 128
N_FFT      = 400
HOP_LEN    = 160
SAMPLE_RATE = 16000

# Special tokens (from qwen3-asr-rs/src/inference.rs)
TOK_IM_START  = 151644   # <|im_start|>
TOK_IM_END    = 151645   # <|im_end|>
TOK_NEWLINE   = 198      # \n
TOK_SYSTEM    = 8948     # system
TOK_USER      = 872      # user
TOK_ASSISTANT = 77091    # assistant
AUDIO_BOS     = 151669   # <|audio_bos|>
AUDIO_EOS_T   = 151670   # <|audio_eos|>
AUDIO_TOK     = 151676   # <|AUDIO|>


# ── Mel spectrogram ───────────────────────────────────────────────────────────

def _slaney_mel_filterbank() -> np.ndarray:
    """Return [MEL_BINS, N_FFT//2+1] Slaney-normalised filterbank (matches mel.rs)."""
    n_freqs = N_FFT // 2 + 1
    fmax    = SAMPLE_RATE / 2.0
    f_sp    = 200.0 / 3.0
    minhz   = 1000.0
    minmel  = minhz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz2mel(f):
        return np.where(f < minhz, f / f_sp,
                        minmel + np.log(np.maximum(f, 1e-9) / minhz) / logstep)

    def mel2hz(m):
        return np.where(m < minmel, f_sp * m,
                        minhz * np.exp(logstep * (m - minmel)))

    mel_min, mel_max = hz2mel(0.0), hz2mel(fmax)
    ffreqs = mel2hz(np.linspace(mel_min, mel_max, MEL_BINS + 2))
    afreqs = np.arange(n_freqs, dtype=np.float64) * fmax / (N_FFT / 2)
    fdiff  = np.diff(ffreqs)

    filters = np.zeros((MEL_BINS, n_freqs), dtype=np.float32)
    for i in range(MEL_BINS):
        down = (afreqs - ffreqs[i]) / fdiff[i]
        up   = (ffreqs[i + 2] - afreqs) / fdiff[i + 1]
        filters[i] = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)
        enorm = 2.0 / (ffreqs[i + 2] - ffreqs[i])
        filters[i] *= enorm
    return filters


_MEL_FILTERS: np.ndarray | None = None


def extract_mel(samples: np.ndarray) -> np.ndarray:
    """
    Extract log-mel spectrogram matching the Rust mel.rs implementation.
    Returns: [MEL_BINS, T]  float32
    """
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = _slaney_mel_filterbank()

    # Pad to next multiple of HOP_LEN
    padded_len = ((len(samples) + HOP_LEN - 1) // HOP_LEN) * HOP_LEN
    padded = np.pad(samples, (0, padded_len - len(samples)))

    # Reflection-pad n_fft/2 on each side (matches Rust reflection_pad)
    padded = np.pad(padded, (N_FFT // 2, N_FFT // 2), mode='reflect')

    # Hann window (0.5 * (1 - cos(2π i / (n-1))))
    window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1))).astype(np.float32)

    # STFT power spectrum [n_frames_total, N_FFT//2+1]
    n_frames_total = (len(padded) - N_FFT) // HOP_LEN + 1
    n_frames = max(n_frames_total - 1, 0)   # drop last frame (matches Rust)
    n_freqs  = N_FFT // 2 + 1

    if n_frames == 0:
        return np.zeros((MEL_BINS, 1), dtype=np.float32)

    frames   = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP_LEN, padded.strides[0]),
    ).copy()
    fft_out  = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    power    = (fft_out.real ** 2 + fft_out.imag ** 2).astype(np.float32)  # [n_frames, n_freqs]

    mel = (_MEL_FILTERS @ power.T).astype(np.float32)   # [MEL_BINS, n_frames]

    # Log normalisation + range clamp (matches Rust)
    log_mel = np.log(np.maximum(mel, 1e-10)) / np.log(10.0)
    max_val = log_mel.max()
    log_mel = (np.maximum(log_mel, max_val - 8.0) + 4.0) / 4.0

    return log_mel


# ── Weight-loading helpers ────────────────────────────────────────────────────

def _w(weights, key):
    t = weights.get(key)
    if t is None:
        raise KeyError(f"Weight not found: {key}")
    return t


def _frozen(t):
    """Return parameter with requires_grad=False."""
    return nn.Parameter(t, requires_grad=False)


def _load_linear(weights, prefix, bias=True):
    w   = _w(weights, f'{prefix}.weight')
    b   = weights.get(f'{prefix}.bias') if bias else None
    lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    lin.weight = _frozen(w)
    if b is not None:
        lin.bias = _frozen(b)
    return lin


def _load_layer_norm(weights, prefix, eps=1e-5):
    w  = _w(weights, f'{prefix}.weight')
    b  = weights.get(f'{prefix}.bias')
    ln = nn.LayerNorm(w.shape[0], eps=eps, elementwise_affine=True)
    ln.weight = _frozen(w)
    if b is not None:
        ln.bias = _frozen(b)
    return ln


def _load_conv2d(weights, prefix, stride=2, padding=1):
    w = _w(weights, f'{prefix}.weight')
    b = weights.get(f'{prefix}.bias')
    out_c, in_c, kH, kW = w.shape
    conv = nn.Conv2d(in_c, out_c, (kH, kW), stride=stride, padding=padding, bias=b is not None)
    conv.weight = _frozen(w)
    if b is not None:
        conv.bias = _frozen(b)
    return conv


# ── Audio Encoder ─────────────────────────────────────────────────────────────

class _AudioAttn(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        hd = AUDIO_D_MODEL // AUDIO_HEADS
        self.n_heads = AUDIO_HEADS
        self.hd      = hd
        self.q_proj   = _load_linear(weights, f'{prefix}.q_proj')
        self.k_proj   = _load_linear(weights, f'{prefix}.k_proj')
        self.v_proj   = _load_linear(weights, f'{prefix}.v_proj')
        self.out_proj = _load_linear(weights, f'{prefix}.out_proj')

    def forward(self, x):
        B, T, _ = x.shape
        nh, hd  = self.n_heads, self.hd
        q = self.q_proj(x).reshape(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, nh, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, nh * hd)
        return self.out_proj(out)


class _AudioLayer(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.pre_norm  = _load_layer_norm(weights, f'{prefix}.self_attn_layer_norm')
        self.attn      = _AudioAttn(weights, f'{prefix}.self_attn')
        self.post_norm = _load_layer_norm(weights, f'{prefix}.final_layer_norm')
        self.fc1       = _load_linear(weights, f'{prefix}.fc1')
        self.fc2       = _load_linear(weights, f'{prefix}.fc2')

    def forward(self, x):
        x = x + self.attn(self.pre_norm(x))
        h = self.fc2(F.gelu(self.fc1(self.post_norm(x)), approximate='none'))
        return x + h


def _sinusoidal_pe(max_len, dim):
    half  = dim // 2
    log_t = math.log(10000.0) / (half - 1)
    emb   = torch.zeros(max_len, dim)
    pos   = torch.arange(max_len, dtype=torch.float32)
    inv   = torch.exp(-torch.arange(half, dtype=torch.float32) * log_t)
    ang   = pos.unsqueeze(1) * inv.unsqueeze(0)
    emb[:, :half] = ang.sin()
    emb[:, half:] = ang.cos()
    return emb


class AudioEncoder(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        p = prefix
        self.conv1    = _load_conv2d(weights, f'{p}.conv2d1')
        self.conv2    = _load_conv2d(weights, f'{p}.conv2d2')
        self.conv3    = _load_conv2d(weights, f'{p}.conv2d3')
        # conv_out has no bias in the weights
        self.conv_out = _load_linear(weights, f'{p}.conv_out', bias=False)
        self.layers   = nn.ModuleList([
            _AudioLayer(weights, f'{p}.layers.{i}') for i in range(AUDIO_LAYERS)
        ])
        self.ln_post  = _load_layer_norm(weights, f'{p}.ln_post')
        self.proj1    = _load_linear(weights, f'{p}.proj1')
        self.proj2    = _load_linear(weights, f'{p}.proj2')
        self.register_buffer('pe', _sinusoidal_pe(1500, AUDIO_D_MODEL))

    @torch.no_grad()
    def forward(self, mel_np: np.ndarray) -> torch.Tensor:
        """
        mel_np: [MEL_BINS, T] float32 numpy
        Returns: [N_tokens, AUDIO_OUTPUT] in model dtype
        """
        dtype  = next(self.parameters()).dtype
        device = next(self.parameters()).device

        x = torch.from_numpy(mel_np).to(device=device, dtype=dtype)
        x = x.unsqueeze(0).unsqueeze(0)   # [1, 1, MEL_BINS, T]

        x = F.gelu(self.conv1(x), approximate='none')
        x = F.gelu(self.conv2(x), approximate='none')
        x = F.gelu(self.conv3(x), approximate='none')

        B, C, F_, T_ = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().reshape(B, T_, C * F_)  # [1, T_, 7680]
        x = self.conv_out(x)                                             # [1, T_, 896]

        pos = self.pe[:T_].unsqueeze(0).to(dtype=dtype)
        x   = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.ln_post(x)
        x = F.gelu(self.proj1(x), approximate='none')
        x = self.proj2(x)                  # [1, T_, 1024]
        return x.squeeze(0)                # [N_tokens, 1024]


# ── Text Decoder ──────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    def __init__(self, weight, eps=TEXT_RMS_EPS):
        super().__init__()
        self.w   = weight   # raw tensor, not a Parameter here — we wrap in caller
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.w.float()).to(x.dtype)


class LoRALinear(nn.Module):
    """Frozen base weight + trainable LoRA delta."""
    def __init__(self, w: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        out_f, in_f = w.shape
        self.w      = w          # frozen base (raw tensor)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_f, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = F.linear(x, self.w)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return base + delta.to(base.dtype)


class _FrozenLinear(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        return F.linear(x, self.w)


def _rope_table(seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos, sin each [seq_len, TEXT_HEAD_DIM]."""
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, TEXT_HEAD_DIM, 2, dtype=torch.float32, device=device) / TEXT_HEAD_DIM
    ))
    t   = torch.arange(seq_len, dtype=torch.float32, device=device)
    f   = torch.outer(t, inv_freq)               # [S, head_dim/2]
    cos = torch.cat([f.cos(), f.cos()], dim=-1)  # [S, head_dim]
    sin = torch.cat([f.sin(), f.sin()], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class _TextAttn(nn.Module):
    def __init__(self, weights, prefix, lora_set: set, rank: int, alpha: float):
        super().__init__()
        for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            w = _w(weights, f'{prefix}.{proj}.weight')
            if proj in lora_set:
                setattr(self, proj, LoRALinear(w, rank, alpha))
            else:
                setattr(self, proj, _FrozenLinear(w))

        self.q_norm = _RMSNorm(_w(weights, f'{prefix}.q_norm.weight'))
        self.k_norm = _RMSNorm(_w(weights, f'{prefix}.k_norm.weight'))

    def forward(self, x, cos4d, sin4d, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, TEXT_Q_HEADS, TEXT_HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = q * cos4d + _rotate_half(q) * sin4d
        k = k * cos4d + _rotate_half(k) * sin4d

        # GQA: expand KV heads
        rep = TEXT_Q_HEADS // TEXT_KV_HEADS
        k   = k.repeat_interleave(rep, dim=1)
        v   = v.repeat_interleave(rep, dim=1)

        attn = (q @ k.transpose(-2, -1)) * (TEXT_HEAD_DIM ** -0.5)
        if mask is not None:
            attn = attn + mask.to(attn.dtype)
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

        out = (attn @ v).transpose(1, 2).reshape(B, T, TEXT_Q_HEADS * TEXT_HEAD_DIM)
        return self.o_proj(out)


class _TextLayer(nn.Module):
    def __init__(self, weights, prefix, lora_set, rank, alpha):
        super().__init__()
        self.in_norm   = _RMSNorm(_w(weights, f'{prefix}.input_layernorm.weight'))
        self.attn      = _TextAttn(weights, f'{prefix}.self_attn', lora_set, rank, alpha)
        self.post_norm = _RMSNorm(_w(weights, f'{prefix}.post_attention_layernorm.weight'))
        self.gate = _w(weights, f'{prefix}.mlp.gate_proj.weight')
        self.up   = _w(weights, f'{prefix}.mlp.up_proj.weight')
        self.down = _w(weights, f'{prefix}.mlp.down_proj.weight')

    def forward(self, x, cos4d, sin4d, mask=None):
        x = x + self.attn(self.in_norm(x), cos4d, sin4d, mask)
        n = self.post_norm(x)
        h = F.linear(F.silu(F.linear(n, self.gate)) * F.linear(n, self.up), self.down)
        return x + h


# ── WAV loading ───────────────────────────────────────────────────────────────

def load_wav_f32(path: str) -> np.ndarray:
    with wave.open(str(path), 'rb') as wf:
        n_ch  = wf.getnchannels()
        sw    = wf.getsampwidth()
        rate  = wf.getframerate()
        raw   = wf.readframes(wf.getnframes())

    if sw == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if n_ch > 1:
        s = s.reshape(-1, n_ch).mean(axis=1)

    if rate != SAMPLE_RATE:
        ratio  = SAMPLE_RATE / rate
        new_len = int(len(s) * ratio)
        indices = np.floor(np.arange(new_len) / ratio).astype(int).clip(0, len(s) - 1)
        s = s[indices]

    return s


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt_ids(n_audio: int, tokenizer, language: str = "German") -> tuple[list[int], int]:
    """
    Return (prompt_token_ids, audio_start_pos) — same structure as Rust build_prompt().
    audio_start_pos: index in prompt_token_ids where audio placeholder tokens begin.
    """
    lang_cap  = language[0].upper() + language[1:]
    lang_ids  = tokenizer.encode(f"language {lang_cap}", add_special_tokens=False).ids

    prefix = [TOK_IM_START, TOK_SYSTEM, TOK_NEWLINE, TOK_IM_END, TOK_NEWLINE,
              TOK_IM_START, TOK_USER,   TOK_NEWLINE, AUDIO_BOS]
    audio_start = len(prefix)
    audio_part  = [AUDIO_TOK] * n_audio
    suffix      = [AUDIO_EOS_T, TOK_IM_END, TOK_NEWLINE,
                   TOK_IM_START, TOK_ASSISTANT, TOK_NEWLINE] + lang_ids

    return prefix + audio_part + suffix, audio_start


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}", flush=True)

    model_path = Path(args.model_dir) / "model.safetensors"
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    tok_path = Path(args.model_dir) / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer.json not found: {tok_path}")

    print("Loading weights...", flush=True)
    weights = load_file(str(model_path), device=device)
    # CPU: candle uses F32, so convert for compatibility when testing offline
    if device == "cpu":
        weights = {k: v.float() for k, v in weights.items()}
        dtype = torch.float32

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("tokenizers library not installed — pip install tokenizers")

    tokenizer = Tokenizer.from_file(str(tok_path))

    target_modules = set(args.target_modules.split(","))
    rank  = args.rank
    alpha = args.lora_alpha

    # Audio encoder (frozen)
    print("Building audio encoder...", flush=True)
    audio_enc = AudioEncoder(weights, "thinker.audio_tower").to(device)
    # Convert all float params to target dtype
    audio_enc = audio_enc.to(dtype)
    audio_enc.requires_grad_(False)

    # Text decoder (with LoRA)
    print(f"Building text decoder ({TEXT_LAYERS} layers)...", flush=True)
    embed_w = weights["thinker.model.embed_tokens.weight"]   # [V, H] bfloat16
    norm_w  = weights["thinker.model.norm.weight"]

    layers_list = []
    for i in tqdm(range(TEXT_LAYERS), desc="  Layers"):
        l = _TextLayer(weights, f"thinker.model.layers.{i}", target_modules, rank, alpha)
        layers_list.append(l)
    text_layers = nn.ModuleList(layers_list).to(device)
    final_norm  = _RMSNorm(norm_w.to(device))

    # Collect LoRA parameters; freeze everything else
    for p in text_layers.parameters():
        p.requires_grad_(False)

    lora_params = []
    for layer in text_layers:
        for proj_name in target_modules:
            proj = getattr(layer.attn, proj_name, None)
            if isinstance(proj, LoRALinear):
                proj.lora_a.requires_grad_(True)
                proj.lora_b.requires_grad_(True)
                lora_params += [proj.lora_a, proj.lora_b]

    if not lora_params:
        sys.exit(f"No LoRA parameters created — check --target-modules (tried: {target_modules})")

    n_lora = sum(p.numel() for p in lora_params)
    print(f"LoRA: rank={rank}, alpha={alpha}, targets={sorted(target_modules)}, "
          f"params={n_lora:,}", flush=True)

    # Training data
    pairs_path = Path(args.data_dir) / "pairs.jsonl"
    if not pairs_path.exists():
        sys.exit(f"pairs.jsonl not found at {pairs_path}")
    pairs = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    if not pairs:
        sys.exit("No training pairs found in pairs.jsonl")
    print(f"Training pairs: {len(pairs)}", flush=True)
    if len(pairs) < 30:
        print(f"  Warning: fewer than 30 pairs — risk of overfitting; consider recording more sentences", flush=True)

    # Precompute per-pair audio embeddings once — the encoder is frozen and deterministic,
    # so repeating encode() every epoch wastes time.
    print("Precomputing audio embeddings...", flush=True)
    cached_embs = {}
    usable_pairs = []
    for pair in tqdm(pairs, desc="  Precompute"):
        audio_path = Path(args.data_dir) / "audio" / f"{pair['id']}.wav"
        if not audio_path.exists():
            print(f"  Warning: missing {audio_path}", file=sys.stderr, flush=True)
            continue
        samples = load_wav_f32(str(audio_path))
        mel = extract_mel(samples)
        cached_embs[pair['id']] = audio_enc(mel).cpu()
        usable_pairs.append(pair)

    if not usable_pairs:
        sys.exit("ERROR: no usable training pairs (all audio files missing?)")

    print(f"Usable pairs: {len(usable_pairs)}/{len(pairs)}", flush=True)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        total_loss = 0.0
        count = 0

        for pair in tqdm(usable_pairs, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            audio_embs = cached_embs[pair['id']].to(device)
            n_audio    = audio_embs.shape[0]

            prompt_ids, audio_start = build_prompt_ids(n_audio, tokenizer, args.language)

            # Tokenize target text (raw text, no special tokens)
            text_ids = tokenizer.encode(pair["text"], add_special_tokens=False).ids

            # Full sequence: prompt + \n + text + <|im_end|>
            # The model generates a newline before the transcription (trimmed by parse_asr_output).
            # Labels: -100 for prompt tokens, actual IDs for newline + text + EOS
            target_ids = [TOK_NEWLINE] + text_ids + [TOK_IM_END]
            labels = [-100] * len(prompt_ids) + target_ids

            # Build embedding sequence:
            #   before_emb [audio_start]  +  audio_embs [n_audio]  +  after_emb [rest + target]
            def embed(ids):
                t = torch.tensor(ids, dtype=torch.long, device=device)
                return F.embedding(t, embed_w)   # [len, 1024]  bfloat16

            before_emb = embed(prompt_ids[:audio_start])
            after_ids  = prompt_ids[audio_start + n_audio:] + target_ids
            after_emb  = embed(after_ids)
            # audio_embs already [N, 1024]

            hidden = torch.cat([before_emb,
                                 audio_embs.to(embed_w.dtype),
                                 after_emb], dim=0).unsqueeze(0)   # [1, S, 1024]
            hidden = hidden.to(dtype)

            S = hidden.shape[1]

            # RoPE [1,1,S,HEAD_DIM]
            cos, sin = _rope_table(S, device, dtype)
            cos4d = cos.unsqueeze(0).unsqueeze(0)
            sin4d = sin.unsqueeze(0).unsqueeze(0)

            # Causal mask [1,1,S,S]
            mask = torch.triu(
                torch.full((S, S), float('-inf'), device=device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            # Forward through text decoder layers
            for layer in text_layers:
                hidden = layer(hidden, cos4d, sin4d, mask)

            # LM head (weight-tied to embed_tokens)
            logits = F.linear(final_norm(hidden).squeeze(0), embed_w)  # [S, V]

            label_t = torch.tensor(labels, dtype=torch.long, device=device)

            # Shift: logits[i] predicts label[i+1]
            loss = F.cross_entropy(
                logits[:-1].float(),
                label_t[1:],
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg = total_loss / max(count, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}: avg_loss={avg:.4f} "
              f"({count}/{len(usable_pairs)} pairs)", flush=True)

    # ── Save adapter ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_tensors = {}
    for li, layer in enumerate(text_layers):
        for proj_name in sorted(target_modules):
            proj = getattr(layer.attn, proj_name, None)
            if isinstance(proj, LoRALinear):
                adapter_tensors[f"layers.{li}.self_attn.{proj_name}.lora_a"] = (
                    proj.lora_a.data.contiguous().bfloat16()
                )
                adapter_tensors[f"layers.{li}.self_attn.{proj_name}.lora_b"] = (
                    proj.lora_b.data.contiguous().bfloat16()
                )

    save_file(adapter_tensors, str(out_dir / "adapter_model.safetensors"))

    cfg = {"r": rank, "lora_alpha": alpha, "target_modules": sorted(target_modules)}
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    n_tensors = len(adapter_tensors)
    n_params  = sum(t.numel() for t in adapter_tensors.values())
    print(f"\nAdapter saved to {out_dir}")
    print(f"  {n_tensors} tensors, {n_params:,} parameters", flush=True)
    print("To activate: POST /lora/reload (restart server if needed)", flush=True)


def main():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-ASR-0.6B")
    p.add_argument("--data-dir",        required=True,  help="Training data directory (pairs.jsonl + audio/)")
    p.add_argument("--model-dir",       required=True,  help="Qwen3-ASR model directory (model.safetensors)")
    p.add_argument("--output-dir",      required=True,  help="Output directory for adapter_model.safetensors")
    p.add_argument("--language",        default="German", help="Language name for prompt prefix (default: German)")
    p.add_argument("--rank",            type=int,   default=8,    help="LoRA rank (default: 8)")
    p.add_argument("--lora-alpha",      type=float, default=8.0,  help="LoRA alpha (default: 8; scale=alpha/rank, so 8/8=1.0)")
    p.add_argument("--lr",              type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    p.add_argument("--epochs",          type=int,   default=3,    help="Training epochs (default: 3)")
    p.add_argument("--target-modules",  default="q_proj,v_proj",
                   help="Comma-separated attention projections to adapt (default: q_proj,v_proj)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
