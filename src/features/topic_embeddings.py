"""
Split each transcript into fixed-word windows (W=30, step=15) and store one
embedding per window using one or more multilingual sentence transformers, each
model in its own subfolder so downstream analysis can compare them. Fixed word
windows are used because the transcripts are partly unpunctuated; sentence
transformers are used because they place cross-lingual paraphrases near each
other rather than clustering by language.

Outputs (per model + task, under
DATA_PROCESSED/topic_embeddings/<model_slug>/<task>/):
    <caliber_code>_emb.npy   shape (n_windows, hidden_dim), L2-normalized
    <caliber_code>_win.txt   one window per line
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import (DATA_PROCESSED,
                        TRANSCRIPT_PART_CLINICAL_CLEAN,
                        TRANSCRIPT_PART_BDAE_CLEAN,
                        TRANSCRIPT_PART_EMOTION_CLEAN)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS TO RUN — each entry produces a separate embeddings subfolder.
#   name    HuggingFace model id or local path
#   slug    folder-safe name; appears in paths and result CSVs
#   prefix  string prepended to every input (e5 models need "passage: ")
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    {"name": "paraphrase-multilingual-mpnet-base-v2",
     "slug": "mpnet",
     "prefix": ""},
    {"name": "paraphrase-multilingual-MiniLM-L12-v2",
     "slug": "minilm",
     "prefix": ""},
    {"name": "intfloat/multilingual-e5-large",
     "slug": "e5_large",
     "prefix": "passage: "},
    {"name": "sentence-transformers/LaBSE",
     "slug": "labse",
     "prefix": ""},
]


TOPIC_EMBEDDING_DIR = Path(DATA_PROCESSED) / "topic_embeddings"

TASK_DIRS = {
    "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_CLEAN,
    "emotion":             TRANSCRIPT_PART_EMOTION_CLEAN,
}


WIN_WORDS = 30
WIN_STEP  = 15
MIN_TAIL  = 10        # drop trailing window shorter than this many words
BATCH     = 32


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: str) -> str:
    """Read a transcript file as text, falling back to windows-1252 when it is
    not valid UTF-8."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding="windows-1252") as f:
            return f.read()


def _windows(text: str, w: int = WIN_WORDS, s: int = WIN_STEP) -> list[str]:
    """Split text into overlapping fixed-word windows of width w advancing by
    step s, returning a single window for short texts and dropping a trailing
    window shorter than MIN_TAIL words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= w:
        return [" ".join(words)] 
    
    out = []
    for start in range(0, len(words), s):
        chunk = words[start:start + w]
        if len(chunk) < MIN_TAIL:
            break
        out.append(" ".join(chunk))
        if start + w >= len(words):
            break
    return out


def _encode(model: SentenceTransformer, texts: list[str],
            prefix: str = "") -> np.ndarray:
    """Encode a list of window strings with the given model, applying the
    optional per-input prefix and returning L2-normalized float32 embeddings."""
    inputs = [prefix + t for t in texts] if prefix else texts
    return model.encode(inputs, batch_size=BATCH,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL EMBEDDING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def embed_one_model(cfg: dict, device: str, overwrite: bool) -> None:
    """Generate and save window embeddings for one model across all configured
    tasks, writing one .npy embedding array and one _win.txt window file per
    transcript and skipping outputs that already exist unless overwrite is set."""
    name, slug, prefix = cfg["name"], cfg["slug"], cfg.get("prefix", "")
    model_root = TOPIC_EMBEDDING_DIR / slug
    model_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  MODEL: {name}")
    print(f"  slug: {slug}   prefix: {repr(prefix)}")
    print("═" * 70)

    model = SentenceTransformer(name, device=device)

    # Record which model produced these embeddings (provenance)
    (model_root / "MODEL.txt").write_text(name, encoding="utf-8")

    for task, mono_dir in TASK_DIRS.items():
        out_dir = model_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in os.listdir(mono_dir) if f.endswith(".txt"))
        print(f"\n[{task}] {len(files)} files → {out_dir}")
        n_done = 0
        for fname in files:
            code = (fname.replace(f"_{task}.txt", "")
                         .replace(".txt", "").strip().lower())
            emb_path = out_dir / f"{code}_emb.npy"
            txt_path = out_dir / f"{code}_win.txt"
            if emb_path.exists() and txt_path.exists() and not overwrite:
                continue
            text = _read_text(os.path.join(mono_dir, fname))
            wins = _windows(text)
            if not wins:
                print(f"  skip empty: {fname}")
                continue
            embeds = _encode(model, wins, prefix=prefix)
            np.save(emb_path, embeds)
            txt_path.write_text("\n".join(wins), encoding="utf-8")
            n_done += 1
        total = len(list(out_dir.glob("*_emb.npy")))
        print(f"  saved {n_done} new (total: {total} patients)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def embedding_generator(overwrite: bool = False) -> None:
    """Entry point: pick a device and run the embedding generation for every
    configured model."""
    TOPIC_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  W={WIN_WORDS}  step={WIN_STEP}  "
          f"models={[m['slug'] for m in MODELS]}")
    for cfg in MODELS:
        embed_one_model(cfg, device, overwrite)


if __name__ == "__main__":
    embedding_generator()