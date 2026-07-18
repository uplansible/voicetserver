#!/usr/bin/env python3
"""Import schmidiscribe training pairs into the unified voicetserver pool.

Unified-server migration (phase 5): voicetserver's pairs pool is canonical and
trains both models' LoRAs. This appends another server's collected pairs
(training/audio/*.wav + training/pairs.jsonl) into it, re-IDing each entry via
the normal max(id)+1 append path so IDs never collide.

Run it on the box holding both data dirs, with voicetserver STOPPED (the
server's pair_write_lock is in-process only — a concurrent upload could
interleave with this append):

    ./tools/import_pairs.py                       # default dirs
    ./tools/import_pairs.py --src ~/.config/schmidiscribe --dest ~/.config/voicetserver
    ./tools/import_pairs.py --dry-run             # show what would be imported

Source WAVs are copied (not moved) — the schmidiscribe pool stays intact until
its retirement. Requires: Python 3 stdlib only.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def load_pairs(jsonl_path: Path) -> list[dict]:
    """Parse a pairs.jsonl file; skip blank/corrupt lines like the server does."""
    pairs = []
    if not jsonl_path.exists():
        return pairs
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  Warning: skipping corrupt line in {jsonl_path}: {line[:60]}…")
    return pairs


def next_id(pairs: list[dict]) -> int:
    """One past the highest numeric id (server semantics — never reuse after delete)."""
    max_id = 0
    for p in pairs:
        try:
            max_id = max(max_id, int(p.get("id", "")))
        except ValueError:
            pass
    return max_id + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Import schmidiscribe training pairs into voicetserver's pool")
    ap.add_argument("--src",  default="~/.config/schmidiscribe",
                    help="Source data dir (contains training/pairs.jsonl + training/audio/) [default: %(default)s]")
    ap.add_argument("--dest", default="~/.config/voicetserver",
                    help="Destination data dir (voicetserver data_dir) [default: %(default)s]")
    ap.add_argument("--dry-run", action="store_true", help="List what would be imported, write nothing")
    args = ap.parse_args()

    src_dir  = Path(args.src).expanduser()
    dest_dir = Path(args.dest).expanduser()
    src_jsonl,  src_audio  = src_dir / "training/pairs.jsonl",  src_dir / "training/audio"
    dest_jsonl, dest_audio = dest_dir / "training/pairs.jsonl", dest_dir / "training/audio"

    src_pairs = load_pairs(src_jsonl)
    if not src_pairs:
        sys.exit(f"No pairs found in {src_jsonl} — nothing to import.")

    # Guard against a running server appending concurrently (PID file check only —
    # the server may still run with a different config dir; stop it to be sure).
    pid_file = Path("~/.config/voicetserver/voicetserver.pid").expanduser()
    if pid_file.exists() and not args.dry_run:
        sys.exit(f"PID file {pid_file} exists — stop voicetserver (voicetserver --stop) before importing.")

    dest_pairs = load_pairs(dest_jsonl)
    new_id = next_id(dest_pairs)
    dest_texts = {p.get("text", "").strip() for p in dest_pairs}

    imported = skipped_dup = skipped_missing = 0
    lines = []
    copies = []  # (src_wav, dest_wav)
    for p in src_pairs:
        text = p.get("text", "").strip()
        src_wav = src_audio / f"{p.get('id', '')}.wav"
        if not src_wav.exists():
            print(f"  Warning: {src_wav} missing — skipping entry '{text[:40]}…'")
            skipped_missing += 1
            continue
        # Same sentence already recorded in the destination pool → likely a
        # calibration sentence recorded on both servers; keep the canonical one.
        if text and text in dest_texts:
            skipped_dup += 1
            continue
        pid = f"{new_id:04}"
        new_id += 1
        copies.append((src_wav, dest_audio / f"{pid}.wav"))
        lines.append(json.dumps(
            {"id": pid, "text": text, "duration_s": round(float(p.get("duration_s", 0.0)), 3)},
            ensure_ascii=False))
        imported += 1

    print(f"Import {imported} pair(s) → {dest_jsonl}"
          f"  (skipped: {skipped_dup} duplicate text, {skipped_missing} missing WAV)")
    if args.dry_run:
        for (s, d), line in zip(copies, lines):
            print(f"  {s} → {d}  {line}")
        return
    if imported == 0:
        return

    dest_audio.mkdir(parents=True, exist_ok=True)
    for s, d in copies:
        shutil.copy2(s, d)
    with dest_jsonl.open("a") as f:
        for line in lines:
            f.write(line + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
