#!/usr/bin/env python3
"""Convert Nemotron 3.5 ASR .nemo into the pure-C runtime binary format.

The generated file is intentionally simple: a little-endian tensor stream with
64-byte aligned float32 payloads, followed by the RNNT vocabulary strings.
Runtime inference does not need Python, PyTorch, NeMo, YAML, or SentencePiece.
"""

from __future__ import annotations

import argparse
import os
import struct
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml


MAGIC = b"NM35ASR\0"
VERSION = 1


def align64(f):
    pad = (-f.tell()) % 64
    if pad:
        f.write(b"\0" * pad)


def extract_member(tar: tarfile.TarFile, suffix: str, dst: Path) -> Path:
    for m in tar.getmembers():
        if m.name.endswith(suffix):
            out = dst / Path(m.name).name
            with tar.extractfile(m) as src, out.open("wb") as fp:
                if src is None:
                    raise RuntimeError(f"cannot extract {suffix}")
                while True:
                    b = src.read(1024 * 1024)
                    if not b:
                        break
                    fp.write(b)
            return out
    raise RuntimeError(f"{suffix} not found in archive")


def load_nemo(nemo_path: Path, work_dir: Path):
    cfg_path = work_dir / "model_config.yaml"
    ckpt_path = work_dir / "model_weights.ckpt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        with tarfile.open(nemo_path, "r:gz") as tar:
            cfg_path = extract_member(tar, "model_config.yaml", work_dir)
            ckpt_path = extract_member(tar, "model_weights.ckpt", work_dir)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    state = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    return cfg, state


def write_model(out_path: Path, cfg: dict, state: dict):
    vocab = list(cfg["joint"]["vocabulary"])
    keys = list(state.keys())
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, len(keys), len(vocab), 0))
        for key in keys:
            tensor = state[key].detach().cpu().contiguous()
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            name = key.encode("utf-8")
            if len(name) > 65535:
                raise ValueError(f"tensor name too long: {key}")
            dims = list(tensor.shape)
            dims4 = dims + [1] * (4 - len(dims))
            raw = tensor.numpy().tobytes(order="C")
            f.write(struct.pack("<HBB", len(name), len(dims), 1))
            f.write(name)
            f.write(struct.pack("<QQQQ", *dims4[:4]))
            f.write(struct.pack("<Q", len(raw)))
            align64(f)
            f.write(raw)
            align64(f)
        f.write(struct.pack("<I", len(vocab)))
        for token in vocab:
            b = str(token).encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo", type=Path, help="nemotron-3.5-asr-streaming-0.6b.nemo")
    ap.add_argument("-o", "--output", type=Path, default=Path("nemotron-3.5-asr-streaming-0.6b.bin"))
    ap.add_argument("--work-dir", type=Path, default=None, help="Reuse/extract into this directory")
    args = ap.parse_args()

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        cfg, state = load_nemo(args.nemo, args.work_dir)
        write_model(args.output, cfg, state)
    else:
        with tempfile.TemporaryDirectory(prefix="nemotron_nemo_") as td:
            cfg, state = load_nemo(args.nemo, Path(td))
            write_model(args.output, cfg, state)

    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"wrote {args.output} ({size_gb:.2f} GiB)")


if __name__ == "__main__":
    main()
