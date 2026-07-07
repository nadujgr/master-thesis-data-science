"""
Lexical baseline features.

Computes purely lexical (surface-form) features from patient speech transcripts
and tests whether they distinguish mood states. Reads per-task transcript files,
extracts token-level statistics (word count, type-token ratio, hapax ratio, mean
word length, long/short word ratios, sequential repetition), merges them with
clinical mood labels, and runs the shared per-task and cross-task consistency
analysis. All features are language-agnostic; language enters the statistical
models as a covariate. Inputs are the monologue transcript directories and the
clinical CSV; outputs are the per-task, consistency, patient-feature and summary
files under results/lexical/ (or the longitudinal equivalent).
"""
# ── Standard library ──
import json
import logging
import re
import string
import sys
import warnings
from pathlib import Path

# ── Third-party ──
import pandas as pd

# ── Local ──
from src.stats_helpers import (
    analyze_experiment,
    consistency_across_experiments,
    fold_mixed,
    print_consistency_summary,
    print_experiment_summary,
)
from src.utils import json_safe as _json_safe
from src.utils import parse_code as _parse_code
from src.utils import read_text_with_fallback
from src.config import (
    AUDIO_DATA_UNIQUE,
    RESULTS_DIR,
    TRANSCRIPT_BY_PART_MONOLOGUE,
    AUDIO_DATA_LONGITUDINAL,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_NAME = "lexical"
OUTPUT_DIR = Path(RESULTS_DIR) / LEVEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONOLOGUE_ROOT = Path(TRANSCRIPT_BY_PART_MONOLOGUE)
CLINICAL_PATH = Path(AUDIO_DATA_UNIQUE)

# All available tasks
TASK_DIRS = {
    "emotion": "emotion_clean",
    "clinical_evaluation": "clinical_evaluation_clean",
    "bdae": "bdae_clean",
}

# Minimum word count for valid transcript
MIN_WORD_COUNT = 10

FEATURE_COLS = [
    "word_count",
    "type_token_ratio",     # unique tokens / total tokens
    "hapax_ratio",          # words appearing once / total
    "mean_word_length",     # mean characters per token
    "long_word_ratio",      # proportion of tokens >=7 chars
    "short_word_ratio",     # proportion of tokens <=3 chars
    "repetition_rate",      # consecutive identical tokens / bigrams
]

_ANON_RE = re.compile(r"\[[^\]]{1,30}\]")
_SENT_END = re.compile(r"[.!?]+")


# ── Helper functions ──


def _read_text(fp: Path) -> str:
    """Read a transcript file, trying utf-8, utf-8-sig, windows-1252 and
    latin-1 so files saved under different locales all read without error."""
    return read_text_with_fallback(
        fp, encodings=("utf-8", "utf-8-sig", "windows-1252", "latin-1")
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


def extract_lexical_features(text: str) -> dict:
    """Extract purely lexical (surface-form) features from raw text."""
    text = _ANON_RE.sub(" ", text).strip()

    if not text:
        return {c: float("nan") for c in FEATURE_COLS}

    words_raw = text.split()
    words = [w.strip(string.punctuation).lower() for w in words_raw]
    words = [w for w in words if w]
    n_words = len(words)
    
    if n_words < MIN_WORD_COUNT:
        return {c: float("nan") for c in FEATURE_COLS}
    
    sentences = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    mean_sent_len = n_words / n_sentences

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    ttr = len(freq) / n_words
    hapax = sum(1 for v in freq.values() if v == 1) / n_words

    mean_wlen = sum(len(w) for w in words) / n_words
    long_ratio = sum(1 for w in words if len(w) >= 7) / n_words
    short_ratio = sum(1 for w in words if len(w) <= 3) / n_words

    punct_rate = sum(1 for ch in text if ch in string.punctuation) / n_words

    bigrams = list(zip(words[:-1], words[1:]))
    repetition_rate = (
        sum(1 for a, b in bigrams if a == b) / len(bigrams) if bigrams else 0.0
    )
    
    return {
        "word_count": n_words,
        "type_token_ratio": ttr,
        "hapax_ratio": hapax,
        "mean_word_length": mean_wlen,
        "long_word_ratio": long_ratio,
        "short_word_ratio": short_ratio,
        "repetition_rate": repetition_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def build_task_features(
    task_name: str,
    task_dir_name: str,
    clinical: pd.DataFrame,
    longitudinal: bool = False,
) -> pd.DataFrame | None:
    """Build feature dataframe for one task."""
    
    task_dir = MONOLOGUE_ROOT / task_dir_name
    if not task_dir.exists():
        log.warning(f"  Task directory not found: {task_dir}")
        return None
    
    records = []
    skipped_short = 0
    
    for fp in sorted(task_dir.glob("*.txt")):
        code = _parse_code(fp.name)
        text = _read_text(fp)

        if not text or len(text.split()) < MIN_WORD_COUNT:
            skipped_short += 1
            continue

        feats = extract_lexical_features(text)
        feats["caliber_code"] = code
        feats["task"] = task_name
        records.append(feats)
    
    if not records:
        log.warning(f"  No valid transcripts found for {task_name}")
        return None
    
    if skipped_short:
        log.info(f"  Skipped {skipped_short} transcripts (too short)")
    
    df = pd.DataFrame(records)
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()

    clinical_cols = [
        c
        for c in [
            "caliber_code",
            "symptom_state",
            "ymrs",
            "hdrs",
            "language",
            "sex",
            "age",
            "visit"
        ]
        if c in clinical.columns
    ]
    
    n_before = len(df)
    merged = df.merge(clinical[clinical_cols], on="caliber_code", how="inner")
    merged = fold_mixed(merged)
    n_after = len(merged)
    
    if n_after < n_before:
        log.info(f"  ⚠ Dropped {n_before - n_after} patients (code not in clinical data)")

    states = merged["symptom_state"].value_counts()
    log.info(f"  [{task_name}] {len(merged)} patients")
    log.info(
        f"    States: "
        + " | ".join(f"{s}={states.get(s, 0)}" for s in ["mania", "depression", "euthymia"])
    )
    
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_lexical(
    tasks: list[str] | None = None,
    longitudinal: bool = False,
    extract_only: bool = False,
) -> None:
    """Run the lexical baseline for the requested tasks: extract features, save
    the patient (or session) feature table, and, in cross-sectional mode, run the
    per-task analysis and cross-task consistency summary. Longitudinal mode only
    saves session features and defers statistics to the longitudinal script."""

    tasks = tasks or list(TASK_DIRS.keys())

    clinical_path = Path(AUDIO_DATA_LONGITUDINAL if longitudinal else AUDIO_DATA_UNIQUE)
    output_dir    = Path(RESULTS_DIR) / ("longitudinal/lexical" if longitudinal
                                          else "lexical")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "LONGITUDINAL" if longitudinal else "CROSS-SECTIONAL"
    log.info("=" * 70)
    log.info(f"  LEVEL 0 — LEXICAL BASELINE ({mode})")
    log.info(f"  Tasks:   {', '.join(tasks)}")
    log.info(f"  Output:  {output_dir}")
    log.info("=" * 70)

    clinical = pd.read_csv(clinical_path)
    clinical.columns = clinical.columns.str.strip().str.lower()
    clinical["caliber_code"] = (
        clinical["caliber_code"].astype(str).str.strip().str.lower()
    )

    log.info("\nExtracting features...")
    task_dfs: dict[str, pd.DataFrame] = {}
    patient_rows: list[pd.DataFrame]  = []

    for task_name in tasks:
        task_dir_name = TASK_DIRS.get(task_name)
        if not task_dir_name:
            log.warning(f"  Unknown task: {task_name}")
            continue

        log.info(f"\n[{task_name}]")
        df = build_task_features(task_name, task_dir_name, clinical,
                                  longitudinal=longitudinal)

        if df is not None and len(df) >= 6:
            task_dfs[task_name] = df
            patient_rows.append(df)

    if not task_dfs:
        log.error("\n✗ No tasks produced sufficient data.")
        return

    if patient_rows:
        patient_df = pd.concat(patient_rows, ignore_index=True)
        patient_df["experiment"] = patient_df["task"].astype(str)

        if longitudinal:
            # all visits per patient
            out_csv = output_dir / "lexical_sessions.csv"
            patient_df.to_csv(out_csv, index=False)
            log.info(f"\n  Sessions saved: {out_csv}")
            log.info(f"    {len(patient_df)} sessions, "
                     f"{patient_df['caliber_code'].nunique()} patients")
        else:
            out_csv = output_dir / "lexical_patient_features.csv"
            patient_df.to_csv(out_csv, index=False)
            log.info(f"\n  Patient features saved: {out_csv}")

    if longitudinal:
        log.info("  longitudinal=True — features saved, skipping cross-sectional analysis.")
        log.info("  Run analysis/longitudinal/longitudinal_analysis.py for longitudinal statistics.")
        return

    log.info("\n" + "=" * 70)
    log.info("  PER-TASK ANALYSIS")
    log.info("=" * 70)

    all_results = []

    for task_name, df in task_dfs.items():
        log.info(f"\n{task_name.upper()}")
        log.info("─" * 60)

        metric_cols = [c for c in FEATURE_COLS if c in df.columns and df[c].nunique() > 1]

        if len(df) < 10 or not metric_cols:
            log.warning("  Insufficient data for analysis")
            continue

        res = analyze_experiment(df, metric_cols)

        if not res.empty:
            res.insert(0, "experiment", task_name)
            res["task"] = task_name
            all_results.append(res)
            print_experiment_summary(task_name, res)

    if not all_results:
        log.error("\n✗ No experiments produced results.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_dir / "lexical_per_task.csv", index=False)

    log.info("\n" + "=" * 70)
    log.info("  CROSS-TASK CONSISTENCY")
    log.info("=" * 70)

    consistency_df = consistency_across_experiments(results_df)
    print_consistency_summary(consistency_df)
    consistency_df.to_csv(output_dir / "lexical_consistency.csv", index=False)

    n_robust_strict = int(consistency_df["robust_strict"].sum())
    n_robust_pole   = int(consistency_df["robust_pole"].sum())

    log.info(f"\n{'═'*70}")
    log.info("  SUMMARY")
    log.info(f"{'═'*70}")
    log.info(f"  Tasks analyzed:                          {len(task_dfs)}")
    log.info(f"  Features tested:                         {len(consistency_df)}")
    log.info(f"  Robust (strict, exact order ≥70%):       {n_robust_strict}")
    log.info(f"  Robust (pole-based, pattern ≥60%):       {n_robust_pole}")

    summary = {
        "mode":             mode,
        "n_tasks":          len(task_dfs),
        "tasks":            list(task_dfs.keys()),
        "n_per_task":       {t: len(d) for t, d in task_dfs.items()},
        "n_features":       len(FEATURE_COLS),
        "n_robust_strict":  n_robust_strict,
        "n_robust_pole":    n_robust_pole,
        "robust_strict":    consistency_df[consistency_df["robust_strict"]]["metric"].tolist(),
        "robust_pole":      consistency_df[consistency_df["robust_pole"]]["metric"].tolist(),
    }

    json_path = output_dir / "lexical_results.json"
    with json_path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    log.info(f"\n  All outputs saved to: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 0 - Lexical Baseline (Task-Agnostic)"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_DIRS.keys()),
        action="append",
        default=None,
        dest="tasks",
        help="Task(s) to process. Default: all.",
    )

    parser.add_argument(
        "--longitudinal",
        action="store_true",
        default=False,
        help="Extract features for longitudinal dataset only. "
             "Statistical analysis is handled by analysis/longitudinal/longitudinal_analysis.py.",
    )
    
    args = parser.parse_args()
    tasks = args.tasks or list(TASK_DIRS.keys())
    
    run_lexical(tasks, longitudinal=args.longitudinal)