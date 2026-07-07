"""
Distributional semantic coherence of patient transcripts at two scales.

Method A (pairwise within-window) is applied to token-level embeddings
(XLM-RoBERTa primary, fastText baseline): at each position the mean cosine
similarity over all word pairs in a window measures local semantic density,
replicating Arslan et al. (2025). Method B (consecutive-window drift) is applied
to pre-computed ~30-word sentence-transformer embeddings (MPNet, MiniLM, E5-large,
LaBSE): cosine similarity between successive windows measures discourse-level
continuity. Both methods summarise each transcript with four features (sem_mean,
sem_std, sem_min, sem_max). An optional stopword-filtering experiment runs the
token-level analysis with and without function-word removal (pronouns kept).

Inputs are the transcript directories, pre-computed embeddings and the clinical
metadata CSV; outputs (per-experiment stats, consistency summaries, per-patient
features and a fastText OOV log) are written under RESULTS_DIR/semantic_similarity.
In longitudinal mode (longitudinal=True) the longitudinal clinical file is used,
outputs go to the longitudinal directory, a session-level CSV is produced for s11,
and the stopword experiment and cross-experiment consistency are skipped.
"""
from __future__ import annotations

import gc
import os
import re
import sys
import warnings
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from src.config import (
    AUDIO_DATA_UNIQUE,
    AUDIO_DATA_LONGITUDINAL,
    DATA_PROCESSED,
    XLM_ROBERTA_DIR,
    FASTTEXT_LANG_MODELS,
    FASTTEXT_DEFAULT_LANG,
    TRANSCRIPT_PART_EMOTION_EMBEDDING,
    TRANSCRIPT_PART_CLINICAL_EMBEDDING,
    RESULTS_DIR,
)
from src.stats_helpers import (
    analyze_experiment,
    consistency_across_experiments,
    fold_mixed,
    print_experiment_summary,
    print_consistency_summary,
    STATES,
)
from src.utils import read_text_with_fallback as _read_text
from src.utils import clean_caliber_code_str as _clean_code

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_NAME = "semantic_similarity"
OUT_DIR    = Path(RESULTS_DIR) / LEVEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Longitudinal output directory (populated only when longitudinal=True)
LONG_OUT_DIR = Path(RESULTS_DIR) / "longitudinal" / LEVEL_NAME

TOPIC_EMBEDDING_DIR        = Path(DATA_PROCESSED) / "topic_embeddings"
SENTENCE_TRANSFORMER_SLUGS = ["mpnet", "minilm", "e5_large", "labse"]

# Window sizes for token-level (Method A: pairwise within-window)
TOKEN_WINDOW_SIZES = [5, 7, 9]
MIN_TOKENS         = 20          # skip transcripts shorter than this

LANG_TO_FT: dict[str, str] = {
    "spanish": "es",
    "catalan": "ca",
    "english": "en",
    "mix":     FASTTEXT_DEFAULT_LANG,
}

TASK_DIRS = {
    "emotion":             TRANSCRIPT_PART_EMOTION_EMBEDDING,
    "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_EMBEDDING,
}

FEATURE_COLS = ["sem_mean", "sem_std", "sem_min", "sem_max"]


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD FILTERING
# ══════════════════════════════════════════════════════════════════════════════

# Pronouns to KEEP (clinically informative)
_PRONOUNS = {
    # English
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    # Spanish
    "yo", "mi", "mí", "me", "nosotros", "nosotras", "nuestro", "nuestra",
    "tú", "te", "ti", "tu", "usted", "él", "ella", "ellos", "ellas",
    "su", "sus", "le", "lo", "la", "los", "las", "se", "sí",
    # Catalan
    "jo", "em", "nosaltres", "nostre", "nostra",
    "tu", "et", "vostè", "ell", "ella", "ells", "elles",
    "seu", "seva", "seus", "seves", "es",
}

# Core stopwords to REMOVE (high-frequency function words)
_STOPWORDS = {
    # Spanish articles
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    # Spanish conjunctions
    'y', 'e', 'o', 'u', 'pero', 'mas', 'sino', 'que', 'porque', 'pues',
    'como', 'cuando', 'donde', 'aunque', 'si', 'ni',
    # Spanish prepositions
    'en', 'de', 'a', 'con', 'por', 'para', 'sin', 'sobre', 'tras',
    'desde', 'hasta', 'hacia', 'entre', 'durante', 'mediante',
    # Spanish auxiliary/copula
    'es', 'está', 'son', 'están', 'hay', 'ser', 'estar', 'haber',
    'era', 'fueron', 'fue', 'sido', 'siendo',
    # Spanish discourse markers
    'pues', 'bueno', 'vale', 'entonces', 'ahora', 'así', 'además',
    'también', 'tampoco', 'sólo', 'solo', 'ya', 'aún', 'aun',
    # Spanish quantifiers/determiners
    'todo', 'toda', 'todos', 'todas', 'mucho', 'mucha', 'muchos', 'muchas',
    'poco', 'poca', 'pocos', 'pocas', 'más', 'menos', 'muy', 'tan',
    'tanto', 'tanta', 'otros', 'otras', 'otro', 'otra',
    # Spanish negation (NOT pronouns)
    'no', 'nada', 'nadie', 'ninguno', 'ninguna',

    # Catalan articles
    'els', 'les', 'uns', 'unes',
    # Catalan conjunctions
    'i', 'però', 'perquè', 'com', 'quan', 'on', 'encara', 'que',
    # Catalan prepositions
    'amb', 'per', 'sense', 'cap', 'des', 'fins', 'cap',
    # Catalan auxiliary/copula
    'és', 'són', 'hi', 'ha', 'haver', 'ser', 'estar',
    # Catalan discourse markers
    'doncs', 'aleshores', 'així', 'també', 'tampoc', 'ja',
    # Catalan quantifiers/determiners
    'tot', 'tota', 'tots', 'totes', 'molt', 'molta', 'molts', 'moltes',
    'poc', 'poca', 'pocs', 'poques', 'més', 'menys', 'tan',
    'altre', 'altra', 'altres',
    # Catalan negation
    'res', 'ningú',

    # English articles
    'the', 'a', 'an',
    # English conjunctions
    'and', 'or', 'but', 'if', 'because', 'as', 'when', 'where', 'while',
    # English prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'about', 'into', 'through', 'during', 'before', 'after',
    # English auxiliary/copula
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    # English discourse markers
    'so', 'then', 'now', 'well', 'also', 'too', 'just', 'only', 'very',
    # English quantifiers/determiners
    'all', 'some', 'any', 'much', 'many', 'more', 'most', 'other',
    # English negation
    'not', 'no', 'nothing', 'nobody', 'none',
}


def filter_stopwords(text: str, remove_stopwords: bool = False) -> str:
    """Remove high-frequency function words while keeping pronouns. Returns the
    original text unchanged when remove_stopwords is False."""
    if not remove_stopwords:
        return text

    words = text.lower().split()
    filtered = []

    for word in words:
        clean_word = word.strip('.,;:!?¿¡"\'()[]{}')
        # Keep pronouns (clinically important) and any non-stopword
        if clean_word in _PRONOUNS or clean_word not in _STOPWORDS:
            filtered.append(word)

    return ' '.join(filtered)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, or 0.0 if either has
    zero norm."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _sim_features(sims: np.ndarray) -> dict:
    """Summarise a 1-D similarity array into four scalar features."""
    if len(sims) == 0:
        return {c: np.nan for c in FEATURE_COLS}
    return {
        "sem_mean": float(np.mean(sims)),
        "sem_std":  float(np.std(sims)),
        "sem_min":  float(np.min(sims)),
        "sem_max":  float(np.max(sims)),
    }


def _analyse(df: pd.DataFrame, exp_name: str) -> pd.DataFrame | None:
    """Run analyze_experiment on FEATURE_COLS; print summary; return result."""
    available = [c for c in FEATURE_COLS
                 if c in df.columns and df[c].nunique() > 1]
    if len(df) < 10 or not available:
        return None
    res = analyze_experiment(df, available)
    if res.empty:
        return None
    res.insert(0, "experiment", exp_name)
    print_experiment_summary(exp_name, res)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# METHOD A — PAIRWISE WITHIN-WINDOW (token-level; replicates Arslan et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

def _pairwise_window_sims(token_vecs: np.ndarray, w: int) -> np.ndarray:
    """Method A: for each window position, the mean cosine similarity over all
    word pairs in [i, i+W). Returns one value per window position. Vectors are
    L2-normalised once so cosines reduce to dot products."""
    n = len(token_vecs)
    if n < w:
        return np.array([])

    norms = np.linalg.norm(token_vecs, axis=1, keepdims=True)
    vecs  = token_vecs / np.where(norms > 0, norms, 1.0)

    sims = []
    for i in range(n - w + 1):
        win  = vecs[i : i + w]
        gram = win @ win.T            # all pairwise cosines at once
        idx  = np.triu_indices(w, k=1)
        sims.append(float(gram[idx].mean()))
    return np.array(sims)


# ── fastText (Method A baseline) ─────────────────────────────────────────────

def _embed_tokens_fasttext(text: str, ft_model) -> tuple[np.ndarray, float]:
    """Return (token_vecs, oov_rate); zero-norm (OOV) tokens are dropped."""
    words = text.lower().split()
    if not words:
        return np.zeros((0, ft_model.get_dimension())), 1.0
    vecs  = np.array([ft_model.get_word_vector(w) for w in words])
    norms = np.linalg.norm(vecs, axis=1)
    return vecs[norms > 0], float(np.mean(norms == 0))


def build_fasttext_features(
    task: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    remove_stopwords: bool = False,
    longitudinal: bool = False,
) -> tuple[dict[int, pd.DataFrame], list[dict]]:
    """Method A applied to fastText static word vectors, processing one language
    at a time to limit memory. In longitudinal mode df_clinical holds one row per
    session; the caliber_code merge is unchanged since each session code is
    unique."""
    transcript_dir = Path(TASK_DIRS[task])
    if not transcript_dir.exists():
        print(f"  ✗ transcript dir not found: {transcript_dir}")
        return {}, []

    valid_codes = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())

    lang_lookup: dict[str, str] = {}
    if "language" in df_clinical.columns:
        lang_lookup = (df_clinical.set_index("caliber_code")["language"]
                       .dropna().str.lower().str.strip().to_dict())

    files_by_lang: dict[str, list[Path]] = {}
    skipped_not_in_csv = 0

    for fpath in sorted(transcript_dir.glob("*.txt")):
        code = _clean_code(fpath.stem, task)

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        raw_lang = lang_lookup.get(code)
        lang_key = LANG_TO_FT.get(str(raw_lang or "").strip().lower(),
                                   FASTTEXT_DEFAULT_LANG)
        files_by_lang.setdefault(lang_key, []).append(fpath)

    if skipped_not_in_csv:
        print(f"  Skipped {skipped_not_in_csv} files not in clinical data")

    rows_by_w: dict[int, list] = {w: [] for w in TOKEN_WINDOW_SIZES}
    oov_log: list[dict] = []
    skipped = 0

    for lang_key, fpaths in files_by_lang.items():
        print(f"  Processing {len(fpaths)} files in {lang_key}...")

        path = FASTTEXT_LANG_MODELS.get(
            lang_key, FASTTEXT_LANG_MODELS[FASTTEXT_DEFAULT_LANG])
        print(f"    Loading fastText [{lang_key}] from {path} …")
        ft_model = fasttext.load_model(str(path))

        for fpath in fpaths:
            code = _clean_code(fpath.stem, task)
            text_raw = _read_text(fpath)
            if not text_raw:
                skipped += 1
                continue

            text = filter_stopwords(text_raw, remove_stopwords)

            token_vecs, oov_rate = _embed_tokens_fasttext(text, ft_model)

            patient_lang = lang_lookup.get(code)

            oov_log.append({
                "caliber_code": code,
                "task": task,
                "language": patient_lang,
                "fasttext_lang": lang_key,
                "oov_rate": oov_rate,
                "word_count_raw": len(text_raw.split()),
                "word_count_filtered": len(text.split()),
                "n_valid_tokens": len(token_vecs),
                "stopwords_removed": remove_stopwords,
            })

            if len(token_vecs) < MIN_TOKENS:
                skipped += 1
                continue

            wc = len(text.split())
            for w in TOKEN_WINDOW_SIZES:
                sims = _pairwise_window_sims(token_vecs, w)
                source_suffix = "_nostop" if remove_stopwords else ""
                row = {
                    "caliber_code": code,
                    "word_count": wc,
                    "word_count_raw": len(text_raw.split()),
                    "n_tokens": len(token_vecs),
                    "oov_rate": oov_rate,
                    "source": f"fasttext_w{w}{source_suffix}",
                    "task": task,
                    "n_windows": len(sims),
                    "stopwords_removed": remove_stopwords,
                }
                row.update(_sim_features(sims))
                rows_by_w[w].append(row)

        del ft_model
        gc.collect()
        print(f"    Cleared {lang_key} model from memory")

    if skipped:
        print(f"  fastText skipped {skipped} transcripts")

    out: dict[int, pd.DataFrame] = {}
    for w, rows in rows_by_w.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)

        df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
        n_before = len(df)
        df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
        if not longitudinal:
            df = fold_mixed(df)
        n_after = len(df)

        if n_after < n_before:
            print(f"    ⚠ Dropped {n_before - n_after} patients (code not in clinical data)")

        suffix = " [no stopwords]" if remove_stopwords else ""
        mode_tag = " [LONG]" if longitudinal else ""
        print(f"  [fasttext w={w}{suffix}{mode_tag}] {task}: {len(df)} rows  "
              f"OOV avg={df['oov_rate'].mean():.2f}")
        out[w] = df

    return out, oov_log


# ── XLM-RoBERTa (Method A, primary) ──────────────────────────────────────────

_xlm_tokenizer = None
_xlm_model     = None


def _load_xlm() -> None:
    """Lazily load the XLM-RoBERTa tokenizer and model into module globals on
    first use, in eval mode."""
    global _xlm_tokenizer, _xlm_model
    if _xlm_tokenizer is None:
        print("  Loading XLM-RoBERTa …")
        _xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_ROBERTA_DIR)
        _xlm_model     = XLMRobertaModel.from_pretrained(XLM_ROBERTA_DIR)
        _xlm_model.eval()


def _embed_tokens_xlm(text: str) -> np.ndarray:
    """Return (n_words, hidden_dim): one vector per word (mean of its subword
    tokens)."""
    enc      = _xlm_tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
    word_ids = _xlm_tokenizer(text, truncation=True,
                               max_length=512).word_ids()
    with torch.no_grad():
        hidden = _xlm_model(**enc).last_hidden_state.squeeze(0).numpy()

    word_vecs: dict[int, list] = {}
    for tok_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        word_vecs.setdefault(word_idx, []).append(hidden[tok_idx])
    return np.array([np.mean(vecs, axis=0)
                     for _, vecs in sorted(word_vecs.items())])


def build_xlm_features(
    task: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    remove_stopwords: bool = False,
    longitudinal: bool = False,
) -> dict[int, pd.DataFrame]:
    """Method A applied to XLM-RoBERTa contextual token embeddings."""
    _load_xlm()
    transcript_dir = Path(TASK_DIRS[task])
    if not transcript_dir.exists():
        print(f"  ✗ transcript dir not found: {transcript_dir}")
        return {}

    valid_codes = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())

    rows_by_w: dict[int, list] = {w: [] for w in TOKEN_WINDOW_SIZES}
    skipped = 0
    skipped_not_in_csv = 0

    for fpath in sorted(transcript_dir.glob("*.txt")):
        code = _clean_code(fpath.stem, task)

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        text_raw = _read_text(fpath)
        text = filter_stopwords(text_raw, remove_stopwords)

        token_vecs = _embed_tokens_xlm(text)
        if len(token_vecs) < MIN_TOKENS:
            skipped += 1
            continue

        wc = len(text.split())
        for w in TOKEN_WINDOW_SIZES:
            sims = _pairwise_window_sims(token_vecs, w)
            source_suffix = "_nostop" if remove_stopwords else ""
            row = {
                "caliber_code": code,
                "word_count": wc,
                "word_count_raw": len(text_raw.split()),
                "n_tokens": len(token_vecs),
                "source": f"xlm_w{w}{source_suffix}",
                "task": task,
                "n_windows": len(sims),
                "stopwords_removed": remove_stopwords,
            }
            row.update(_sim_features(sims))
            rows_by_w[w].append(row)

    if skipped_not_in_csv:
        print(f"  Skipped {skipped_not_in_csv} files not in clinical data")

    out: dict[int, pd.DataFrame] = {}
    for w, rows in rows_by_w.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)

        df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
        n_before = len(df)
        df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
        if not longitudinal:
            df = fold_mixed(df)
        n_after = len(df)

        if n_after < n_before:
            print(f"    ⚠ Dropped {n_before - n_after} patients (code not in clinical data)")

        mode_tag = " [LONG]" if longitudinal else ""
        suffix = " [no stopwords]" if remove_stopwords else ""
        print(f"  [xlm_roberta w={w}{suffix}{mode_tag}] {task}: {len(df)} rows")
        out[w] = df
    return out


# ══════════════════════════════════════════════════════════════════════════════
# METHOD B — CONSECUTIVE WINDOW DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def _consecutive_window_sims(emb: np.ndarray) -> np.ndarray:
    """Method B: cos(e_i, e_{i+1}) for each consecutive window pair, one value per
    transition. emb is (n_windows, dim); vectors are re-normalised and results
    clipped to [-1, 1]."""
    if emb.shape[0] < 2:
        return np.array([])
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    vecs  = emb / np.where(norms > 0, norms, 1.0)
    sims  = np.einsum("ij,ij->i", vecs[:-1], vecs[1:])
    return sims.clip(-1.0, 1.0)


def build_sentence_transformer_features(
    task: str,
    slug: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    longitudinal: bool = False,
) -> pd.DataFrame:
    """Method B applied to pre-computed sentence-transformer embeddings. In
    longitudinal mode, session-level caliber_codes absent from the (cross-
    sectional) embedding directory yield an empty DataFrame plus a warning."""
    emb_dir = TOPIC_EMBEDDING_DIR / slug / task
    if not emb_dir.is_dir():
        return pd.DataFrame()

    valid_codes = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())

    rows: list[dict] = []
    skipped_not_in_csv = 0

    for emb_path in sorted(emb_dir.glob("*_emb.npy")):
        code = emb_path.stem.replace("_emb", "")

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        txt_path = emb_dir / f"{code}_win.txt"
        if not txt_path.exists():
            continue
        emb  = np.load(emb_path).astype(np.float32)
        wins = txt_path.read_text(encoding="utf-8").splitlines()
        if emb.shape[0] != len(wins) or emb.shape[0] < 2:
            continue
        word_count = sum(len(w.split()) for w in wins)
        sims       = _consecutive_window_sims(emb)
        row        = {
            "caliber_code": code,
            "word_count": word_count,
            "n_windows": emb.shape[0]
        }
        row.update(_sim_features(sims))
        rows.append(row)

    if skipped_not_in_csv:
        mode_tag = "longitudinal sessions" if longitudinal else "patients"
        print(f"  Skipped {skipped_not_in_csv} embeddings not matching {mode_tag}")

    if not rows:
        if longitudinal:
            print(f"  ⚠ [{slug}] {task}: no longitudinal session codes found in "
                  f"pre-computed embeddings. Re-run the embedding stage with "
                  f"longitudinal codes to populate Method B for the longitudinal subset.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
    n_before = len(df)
    df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
    if not longitudinal:
        df = fold_mixed(df)
    n_after = len(df)

    if n_after < n_before:
        print(f"    ⚠ Dropped {n_before - n_after} rows (code not in clinical data)")

    mode_tag = " [LONG]" if longitudinal else ""
    print(f"  [{slug}{mode_tag}] {task}: {len(df)} rows, "
          f"{df['n_windows'].mean():.0f} windows/session avg")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LONGITUDINAL SESSION CSV
# ══════════════════════════════════════════════════════════════════════════════

def _build_longitudinal_sessions_csv(
    patient_rows: list[pd.DataFrame],
    out_dir: Path,
) -> None:
    """Aggregate per-session feature rows into the sessions CSV consumed by s11,
    one row per (caliber_code, source, task). Multiple window sizes of the same
    source family (e.g. xlm_w5/w7/w9) are averaged into a single row so s11
    receives one scalar per patient/session/source."""
    if not patient_rows:
        print("  ⚠ No longitudinal session rows to write.")
        return

    df = pd.concat(patient_rows, ignore_index=True)

    # Source family = source with the _w<N> window-size suffix stripped
    def _family(src: str) -> str:
        return re.sub(r"_w\d+", "", src)

    df["source_family"] = df["source"].apply(_family)

    # Average sem_* features across window sizes within each
    # (caliber_code, task, source_family) group, keeping clinical cols
    non_feature_cols = {"source", "source_family", "n_windows", "n_tokens",
                        "oov_rate", "word_count_raw", "stopwords_removed"}
    id_cols = ["caliber_code", "task", "source_family"]
    clinical_pass_cols = [c for c in df.columns
                          if c not in non_feature_cols
                          and c not in FEATURE_COLS
                          and c not in id_cols]

    agg_cols = FEATURE_COLS + (["word_count"] if "word_count" in df.columns else [])
    agg_df = (
        df.groupby(id_cols + clinical_pass_cols, as_index=False)[agg_cols]
        .mean()
    )

    # Rename source_family back to "source" for consistency with other session CSVs
    agg_df = agg_df.rename(columns={"source_family": "source"})

    out_path = out_dir / "semantic_similarity_sessions.csv"
    agg_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Longitudinal sessions CSV → {out_path}")
    print(f"    {len(agg_df)} rows  ·  "
          f"{agg_df['caliber_code'].nunique()} unique session codes  ·  "
          f"{agg_df['source'].nunique()} sources")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_semantic_similarity(
    tasks: list[str] | None = None,
    use_fasttext: bool = True,
    use_sentence_transformers: bool = True,
    use_xlm: bool = True,
    run_stopword_experiment: bool = True,
    longitudinal: bool = False,
) -> None:
    """Run the semantic coherence pipeline over the selected embedding sources.
    run_stopword_experiment runs the token-level analysis with and without
    stopword filtering (auto-disabled when longitudinal=True). In longitudinal
    mode the longitudinal clinical file and output directory are used and a
    sessions CSV is produced for s11."""
    tasks = tasks or list(TASK_DIRS.keys())

    # ── Longitudinal mode overrides ───────────────────────────────────────────
    if longitudinal:
        run_stopword_experiment = False
        out_dir = LONG_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        clinical_csv = AUDIO_DATA_LONGITUDINAL
        mode_label   = "LONGITUDINAL"
    else:
        out_dir      = OUT_DIR
        clinical_csv = AUDIO_DATA_UNIQUE
        mode_label   = "CROSS-SECTIONAL"

    file_suffix = "_stopword_exp" if run_stopword_experiment else ""

    print("═" * 70)
    print(f"  LEVEL 5 — SEMANTIC COHERENCE  [{mode_label}]")
    print("  Method A (token-level): pairwise within-window — replicates Arslan et al. 2025")
    print("  Method B (sentence-level): consecutive window drift")
    sources = [s for s, f in [("fastText [A]", use_fasttext),
                               ("sentence-transformers [B]", use_sentence_transformers),
                               ("XLM-RoBERTa [A]", use_xlm)] if f]
    print(f"  Sources: {', '.join(sources)}")

    if longitudinal:
        print("\n  ⚠  LONGITUDINAL MODE")
        print("     • Clinical data: AUDIO_DATA_LONGITUDINAL")
        print("     • Outputs:       results/longitudinal/semantic_similarity/")
        print("     • Stopword experiment disabled (run cross-sectional pipeline separately)")
        print("     • fold_mixed skipped (s11 handles this via _fold_mixed helper)")
        print("     • Cross-experiment consistency skipped (handled by s11)")

    if run_stopword_experiment:
        print("\n  🧪 STOPWORD EXPERIMENT: Running BOTH with and without function word filtering")
        print("     Hypothesis: XLM floor effect driven by high-frequency function words")

    print("═" * 70)

    # ── Load clinical data ────────────────────────────────────────────────────
    df_clinical = pd.read_csv(clinical_csv)
    df_clinical["caliber_code"] = (
        df_clinical["caliber_code"].astype(str).str.strip().str.lower())
    clinical_cols = [c for c in
                     ["caliber_code", "symptom_state", "language",
                      "sex", "age", "ymrs", "hdrs"]
                     if c in df_clinical.columns]

    all_results:  list[pd.DataFrame] = []
    patient_rows: list[pd.DataFrame] = []
    all_oov_log:  list[dict]         = []

    filter_conditions = [False]
    if run_stopword_experiment:
        filter_conditions.append(True)

    # ── Source 1: fastText — Method A, baseline ───────────────────────────────
    if use_fasttext:
        for remove_sw in filter_conditions:
            suffix = " [STOPWORD EXPERIMENT]" if remove_sw else ""
            print("\n" + "─" * 60)
            print(f"  SOURCE 1: fastText  [Method A — pairwise within-window]{suffix}")
            print("  Baseline: static vectors, language-specific models.")
            print("  Limited cross-lingual alignment (Spanish ≠ Catalan space).")
            print("─" * 60)

            for task in tasks:
                print(f"\n[fasttext{' (no stopwords)' if remove_sw else ''}] {task}")
                dfs_by_w, oov_log = build_fasttext_features(
                    task, df_clinical, clinical_cols,
                    remove_stopwords=remove_sw,
                    longitudinal=longitudinal,
                )
                all_oov_log.extend(oov_log)

                for w, df in dfs_by_w.items():
                    if df.empty:
                        continue
                    patient_rows.append(df)

                    if not longitudinal:
                        exp_suffix = "_nostop" if remove_sw else ""
                        res = _analyse(df, f"fasttext_w{w}{exp_suffix}_{task}")
                        if res is not None:
                            res["source"] = f"fasttext_w{w}{exp_suffix}"
                            res["task"] = task
                            res["stopwords_removed"] = remove_sw
                            all_results.append(res)

    # ── Source 2: sentence transformers — Method B ────────────────────────────
    if use_sentence_transformers:
        print("\n" + "─" * 60)
        print("  SOURCE 2: sentence transformers  [Method B — consecutive drift]")
        print("  Pre-computed 30-word window embeddings (reused from Level 6).")
        print("  Best cross-lingual alignment. Captures discourse continuity.")
        if longitudinal:
            print("  ⚠  In longitudinal mode, embeddings must contain session-level")
            print("     caliber_codes to be matched.  Empty result = codes not found.")
        print("─" * 60)
        for slug in SENTENCE_TRANSFORMER_SLUGS:
            slug_dir = TOPIC_EMBEDDING_DIR / slug
            if not slug_dir.is_dir():
                print(f"  ✗ {slug} not found — run Level 6 embedding stage first")
                continue
            for task in tasks:
                print(f"\n[{slug}] {task}")
                df = build_sentence_transformer_features(
                    task, slug, df_clinical, clinical_cols,
                    longitudinal=longitudinal,
                )
                if df.empty:
                    continue
                df["source"] = slug
                df["task"] = task
                df["stopwords_removed"] = False
                patient_rows.append(df)

                if not longitudinal:
                    res = _analyse(df, f"st_{slug}_{task}")
                    if res is not None:
                        res["source"], res["task"] = slug, task
                        res["stopwords_removed"] = False
                        all_results.append(res)

    # ── Source 3: XLM-RoBERTa — Method A, primary ────────────────────────────
    if use_xlm:
        for remove_sw in filter_conditions:
            suffix = " [STOPWORD EXPERIMENT]" if remove_sw else ""
            print("\n" + "─" * 60)
            print(f"  SOURCE 3: XLM-RoBERTa  [Method A — pairwise within-window]{suffix}")
            print("  Contextual token embeddings, one multilingual model.")
            print("  Primary replication of Arslan et al. (2025).")
            print("─" * 60)

            for task in tasks:
                print(f"\n[xlm_roberta{' (no stopwords)' if remove_sw else ''}] {task}")
                dfs_by_w = build_xlm_features(
                    task, df_clinical, clinical_cols,
                    remove_stopwords=remove_sw,
                    longitudinal=longitudinal,
                )
                for w, df in dfs_by_w.items():
                    if df.empty:
                        continue
                    exp_suffix = "_nostop" if remove_sw else ""
                    df["source"] = f"xlm_w{w}{exp_suffix}"
                    df["task"] = task
                    patient_rows.append(df)

                    if not longitudinal:
                        res = _analyse(df, f"xlm_w{w}{exp_suffix}_{task}")
                        if res is not None:
                            res["source"] = f"xlm_w{w}{exp_suffix}"
                            res["task"] = task
                            res["stopwords_removed"] = remove_sw
                            all_results.append(res)

    # ══════════════════════════════════════════════════════════════════════════
    # LONGITUDINAL: write sessions CSV and exit
    # ══════════════════════════════════════════════════════════════════════════
    if longitudinal:
        _build_longitudinal_sessions_csv(patient_rows, out_dir)

        if all_oov_log:
            pd.DataFrame(all_oov_log).to_csv(
                out_dir / "fasttext_oov_log_longitudinal.csv", index=False)
            print(f"  fastText OOV log → {out_dir / 'fasttext_oov_log_longitudinal.csv'}")

        print(f"\n{'═'*70}")
        print("  LONGITUDINAL SEMANTIC COHERENCE EXTRACTION COMPLETE")
        print(f"  Outputs → {out_dir}")
        print(f"{'═'*70}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-SECTIONAL: consistency + stopword analysis (unchanged below)
    # ══════════════════════════════════════════════════════════════════════════
    if not all_results:
        print("\n✗ No experiments produced results.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(out_dir / f"semantic_per_experiment{file_suffix}.csv", index=False)

    # ── Stopword experiment analysis ──────────────────────────────────────────
    if run_stopword_experiment:
        print("\n" + "═" * 70)
        print("  STOPWORD FILTERING EXPERIMENT RESULTS")
        print("═" * 70)

        if patient_rows:
            patient_df = pd.concat(patient_rows, ignore_index=True)

            xlm_patient_df = patient_df[
                patient_df["source"].str.contains("xlm", case=False)
            ].copy()

            if not xlm_patient_df.empty and "stopwords_removed" in xlm_patient_df.columns:
                print("\n📊 XLM-RoBERTa: Impact of Stopword Removal")
                print("─" * 70)

                for metric in ["sem_mean", "sem_std", "sem_min", "sem_max"]:
                    if metric not in xlm_patient_df.columns:
                        continue

                    with_stop = xlm_patient_df[~xlm_patient_df["stopwords_removed"]]
                    no_stop   = xlm_patient_df[xlm_patient_df["stopwords_removed"]]

                    if len(with_stop) > 0 and len(no_stop) > 0:
                        mean_with = with_stop[metric].mean()
                        mean_no   = no_stop[metric].mean()
                        std_with  = with_stop[metric].std()
                        std_no    = no_stop[metric].std()

                        print(f"\n  {metric}:")
                        print(f"    With stopwords:    {mean_with:.4f} (SD={std_with:.4f})")
                        print(f"    Without stopwords: {mean_no:.4f} (SD={std_no:.4f})")
                        print(f"    Δ (change):        {mean_no - mean_with:+.4f}")

                        if metric == "sem_mean":
                            if mean_no < mean_with - 0.01:
                                print(f"    ✓ Mean similarity DECREASED (more variance in embedding space)")
                            elif mean_no > mean_with + 0.01:
                                print(f"    ✗ Mean similarity INCREASED (unexpected)")
                            else:
                                print(f"    ≈ No substantial change (<1%)")
                        elif metric == "sem_std":
                            if std_no > std_with:
                                print(f"    ✓ Variability INCREASED (more differentiation)")
                            else:
                                print(f"    → Variability unchanged or decreased")

                comparison_rows = []
                for task in xlm_patient_df["task"].unique():
                    for source in xlm_patient_df["source"].unique():
                        if "xlm" not in source.lower():
                            continue

                        base_source = source.replace("_nostop", "")

                        mask_with = (
                            (xlm_patient_df["task"] == task) &
                            (xlm_patient_df["source"] == base_source) &
                            (~xlm_patient_df["stopwords_removed"])
                        )
                        mask_without = (
                            (xlm_patient_df["task"] == task) &
                            (xlm_patient_df["source"] == f"{base_source}_nostop") &
                            (xlm_patient_df["stopwords_removed"])
                        )

                        with_data    = xlm_patient_df[mask_with]
                        without_data = xlm_patient_df[mask_without]

                        if len(with_data) > 0 and len(without_data) > 0:
                            row = {
                                "source":               base_source,
                                "task":                 task,
                                "n_patients_with":      len(with_data),
                                "n_patients_without":   len(without_data),
                            }

                            for metric in ["sem_mean", "sem_std", "sem_min", "sem_max"]:
                                if metric in with_data.columns:
                                    row[f"{metric}_with"]    = with_data[metric].mean()
                                    row[f"{metric}_without"] = without_data[metric].mean()
                                    row[f"{metric}_delta"]   = (
                                        without_data[metric].mean() - with_data[metric].mean()
                                    )

                            comparison_rows.append(row)

                if comparison_rows:
                    stopword_comparison = pd.DataFrame(comparison_rows)
                    stopword_comparison.to_csv(
                        out_dir / f"semantic_stopword_comparison{file_suffix}.csv",
                        index=False
                    )
                    print(f"\n  Stopword comparison → "
                          f"{out_dir / f'semantic_stopword_comparison{file_suffix}.csv'}")

                    print("\n  Summary by window size:")
                    print("  " + "─" * 66)
                    for _, row in stopword_comparison.iterrows():
                        if row["task"] == "emotion":
                            print(f"  {row['source']:20s}  "
                                  f"mean: {row.get('sem_mean_with', 0):.4f} → "
                                  f"{row.get('sem_mean_without', 0):.4f}  "
                                  f"(Δ {row.get('sem_mean_delta', 0):+.4f})")

    # ── Method A consistency ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  METHOD A: Token-Level Pairwise Within-Window")
    print("  Robustness = window stability + task generalizability")
    print("═" * 70)

    methodA_df = results_df[
        results_df["source"].str.contains("xlm|fasttext", case=False, na=False)
    ].copy()

    xlm_df = pd.DataFrame()
    ft_df  = pd.DataFrame()

    if not methodA_df.empty:
        methodA_consistency = consistency_across_experiments(methodA_df)
        print("\n[Combined: XLM-RoBERTa + fastText]")
        print_consistency_summary(methodA_consistency)
        methodA_consistency.to_csv(
            out_dir / f"semantic_consistency_methodA_combined{file_suffix}.csv", index=False)

        xlm_df = methodA_df[methodA_df["source"].str.contains("xlm", case=False)]
        if not xlm_df.empty:
            xlm_consistency = consistency_across_experiments(xlm_df)
            print("\n[XLM-RoBERTa only — PRIMARY]")
            print_consistency_summary(xlm_consistency)
            xlm_consistency.to_csv(
                out_dir / f"semantic_consistency_methodA_xlm{file_suffix}.csv", index=False)

        ft_df = methodA_df[methodA_df["source"].str.contains("fasttext", case=False)]
        if not ft_df.empty:
            ft_consistency = consistency_across_experiments(ft_df)
            print("\n[fastText only — BASELINE]")
            print_consistency_summary(ft_consistency)
            ft_consistency.to_csv(
                out_dir / f"semantic_consistency_methodA_fasttext{file_suffix}.csv", index=False)

    # ── Method B consistency ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  METHOD B: Sentence-Level Consecutive Drift")
    print("  Robustness = model agreement + task generalizability")
    print("═" * 70)

    methodB_df = results_df[
        results_df["source"].isin(SENTENCE_TRANSFORMER_SLUGS)
    ].copy()

    if not methodB_df.empty:
        methodB_consistency = consistency_across_experiments(methodB_df)
        print_consistency_summary(methodB_consistency)
        methodB_consistency.to_csv(
            out_dir / f"semantic_consistency_methodB{file_suffix}.csv", index=False)

    # ── Output summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  OUTPUT FILES")
    print(f"{'═'*70}")
    print(f"  Per-experiment results:          "
          f"{out_dir / f'semantic_per_experiment{file_suffix}.csv'}")
    if run_stopword_experiment:
        print(f"  Stopword experiment comparison:  "
              f"{out_dir / f'semantic_stopword_comparison{file_suffix}.csv'}")
    if not methodA_df.empty:
        print(f"  Method A consistency (combined): "
              f"{out_dir / f'semantic_consistency_methodA_combined{file_suffix}.csv'}")
        if not xlm_df.empty:
            print(f"  Method A consistency (XLM only): "
                  f"{out_dir / f'semantic_consistency_methodA_xlm{file_suffix}.csv'}")
        if not ft_df.empty:
            print(f"  Method A consistency (FT only):  "
                  f"{out_dir / f'semantic_consistency_methodA_fasttext{file_suffix}.csv'}")
    if not methodB_df.empty:
        print(f"  Method B consistency:            "
              f"{out_dir / f'semantic_consistency_methodB{file_suffix}.csv'}")

    if patient_rows:
        patient_df = pd.concat(patient_rows, ignore_index=True)
        src = patient_df["source"].astype(str)
        st_prefix = np.where(src.isin(SENTENCE_TRANSFORMER_SLUGS), "st_", "")
        patient_df["experiment"] = (
            st_prefix + src + "_" + patient_df["task"].astype(str)
        )
        patient_df.to_csv(
            out_dir / f"semantic_patient_features{file_suffix}.csv", index=False)
        print(f"  Patient features:                "
              f"{out_dir / f'semantic_patient_features{file_suffix}.csv'}")
        print(f"    {len(patient_df)} observations, "
              f"{patient_df['caliber_code'].nunique()} patients, "
              f"{patient_df['experiment'].nunique()} experiments")

    if all_oov_log:
        pd.DataFrame(all_oov_log).to_csv(
            out_dir / f"fasttext_oov_log{file_suffix}.csv", index=False)
        print(f"  fastText OOV log:                "
              f"{out_dir / f'fasttext_oov_log{file_suffix}.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Level 5 — Semantic coherence (two complementary methods).")
    parser.add_argument("--task", choices=list(TASK_DIRS.keys()),
                        action="append", default=None, dest="tasks",
                        help="Task(s) to process. Default: all.")
    parser.add_argument("--source",
                        choices=["fasttext", "sentence_transformers", "xlm", "all"],
                        action="append", default=None, dest="sources",
                        help="Embedding source(s) to run. Default: all.")
    parser.add_argument("--stopword-experiment", action="store_true",
                        help="Run parallel analysis with/without stopword filtering.")
    parser.add_argument("--longitudinal", action="store_true",
                        help="Run in longitudinal mode: load AUDIO_DATA_LONGITUDINAL, "
                             "write to results/longitudinal/semantic_similarity/, "
                             "produce semantic_similarity_sessions.csv for s11.")

    args    = parser.parse_args()
    tasks   = args.tasks or list(TASK_DIRS.keys())
    sources = set(args.sources or ["all"])
    use_all = "all" in sources

    run_semantic_similarity(
        tasks,
        use_fasttext              = use_all or "fasttext"              in sources,
        use_sentence_transformers = use_all or "sentence_transformers" in sources,
        use_xlm                   = use_all or "xlm"                   in sources,
        run_stopword_experiment   = args.stopword_experiment,
        longitudinal              = args.longitudinal,
    )