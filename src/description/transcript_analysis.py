"""
Data-section descriptor for cleaned patient monologue transcripts. For the
unique and longitudinal datasets it builds a per-dataset × per-task availability
table (transcripts ≥ MIN_WORD_COUNT, flagging state-asymmetric dropout), per-task
word-count descriptives, and a list of exclusion candidates below the word-count
threshold. Reads the cleaned-monologue folders and the unique/longitudinal
metadata CSVs, and writes transcript_availability.csv, word_count_descriptives.csv
and exclusion_candidates.csv under results/transcript_analysis/.
"""

from __future__ import annotations

import os
import re
import string
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.config import (
    AUDIO_DATA_LONGITUDINAL,
    AUDIO_DATA_UNIQUE,
    RESULTS_DIR,
    TRANSCRIPT_PART_BDAE_CLEAN,
    TRANSCRIPT_PART_CLINICAL_CLEAN,
    TRANSCRIPT_PART_EMOTION_CLEAN,
)

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

LEVEL_NAME      = "transcript_analysis"
RESULTS_SUBDIR  = Path(RESULTS_DIR) / LEVEL_NAME
RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)

MIN_WORD_COUNT  = 30
DROPOUT_FLAG_PP = 15.0

TASK_DIRS: dict[str, Path] = {
    "bdae":                Path(TRANSCRIPT_PART_BDAE_CLEAN),
    "emotion":             Path(TRANSCRIPT_PART_EMOTION_CLEAN),
    "clinical_evaluation": Path(TRANSCRIPT_PART_CLINICAL_CLEAN),
}

ORDER  = ["depression", "euthymia", "mania"]
_ANON_RE = re.compile(r"\[[^\]]{1,30}\]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    """Read a transcript file as text, falling back to windows-1252 when it is
    not valid UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="windows-1252")


def _count_words(text: str) -> int:
    """Count word tokens in a transcript, stripping bracketed anonymisation
    markers and surrounding punctuation before counting."""
    text = _ANON_RE.sub(" ", text).strip()
    if not text:
        return 0
    tokens = [w.strip(string.punctuation).lower() for w in text.split()]
    return sum(1 for w in tokens if w)


def _fold_mixed(s: pd.Series) -> pd.Series:
    """Collapse the "mixed" symptom state into "mania" to match the 3-level
    analysis grouping."""
    return s.astype(str).replace({"mixed": "mania"})


# ── Transcript lookup ─────────────────────────────────────────────────────────

def _index_task_folder(task: str, task_dir: Path) -> dict[str, int]:
    """Return {caliber_code: word_count} for all .txt files in folder."""
    if not task_dir.is_dir():
        print(f"  ⚠ Missing folder for {task}: {task_dir}")
        return {}
    out: dict[str, int] = {}
    suffix = f"_{task}.txt"
    for fn in os.listdir(task_dir):
        if not fn.endswith(".txt"):
            continue
        code = fn.replace(suffix, "").replace(".txt", "").strip().lower()
        out[code] = _count_words(_read_text(task_dir / fn))
    return out


# ── Availability ──────────────────────────────────────────────────────────────

def _build_availability(dataset_name: str,
                        clinical: pd.DataFrame,
                        transcripts: dict[str, dict[str, int]]
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-task availability table for one dataset: count transcripts
    that are present and meet the minimum word count, and flag any task whose
    dropout differs across symptom states by more than DROPOUT_FLAG_PP points."""
    rows:  list[dict] = []
    warns: list[str]  = []
    if len(clinical) == 0:
        return pd.DataFrame(rows), warns

    clinical = clinical.copy()
    clinical["caliber_code"]  = clinical["caliber_code"].astype(str).str.strip().str.lower()
    clinical["symptom_state"] = _fold_mixed(clinical["symptom_state"])
    n_total = len(clinical)

    for task, code_to_wc in transcripts.items():
        present = clinical["caliber_code"].isin(code_to_wc)
        wc_ok   = clinical["caliber_code"].map(code_to_wc).fillna(0).astype(int) >= MIN_WORD_COUNT
        valid   = present & wc_ok
        n_valid = int(valid.sum())
        rows.append({
            "dataset":       dataset_name,
            "task":          task,
            "n_available":   n_valid,
            "n_total":       n_total,
            "pct_available": round(100 * n_valid / n_total, 1),
        })

        # Flag tasks whose dropout differs across states by > DROPOUT_FLAG_PP.
        per_state = (clinical.assign(_missing=~valid)
                             .groupby("symptom_state")["_missing"]
                             .agg(["sum", "count"]))
        per_state["pct"] = 100 * per_state["sum"] / per_state["count"]
        per_state = per_state[per_state["count"] >= 3]
        if not per_state.empty:
            spread = per_state["pct"].max() - per_state["pct"].min()
            if spread > DROPOUT_FLAG_PP:
                breakdown = ", ".join(
                    f"{st}={int(r['sum'])}/{int(r['count'])} ({r['pct']:.0f}%)"
                    for st, r in per_state.iterrows()
                )
                warns.append(
                    f"[{dataset_name} / {task}] dropout differs by "
                    f"{spread:.0f} pp across states — {breakdown}"
                )
    return pd.DataFrame(rows), warns


# ── Word-count descriptives ───────────────────────────────────────────────────

def _build_descriptives(dataset_name: str,
                        clinical: pd.DataFrame,
                        transcripts: dict[str, dict[str, int]]
                        ) -> pd.DataFrame:
    """Compute per-task word-count descriptives (n, count below threshold, min,
    max, median, mean, SD) over all found transcripts for one dataset."""
    rows: list[dict] = []
    for task, code_to_wc in transcripts.items():
        wc_values = list(code_to_wc.values())
        if not wc_values:
            continue
        rows.append({
            "dataset":             dataset_name,
            "task":                task,
            "n_found":             len(wc_values),
            "n_below_threshold":   sum(1 for w in wc_values if w < MIN_WORD_COUNT),
            "min":                 int(min(wc_values)),
            "max":                 int(max(wc_values)),
            "median":              round(float(pd.Series(wc_values).median()), 1),
            "mean":                round(float(pd.Series(wc_values).mean()), 1),
            "sd":                  round(float(pd.Series(wc_values).std()), 1),
        })
    return pd.DataFrame(rows)


# ── Exclusion candidates ──────────────────────────────────────────────────────

def _build_exclusion_candidates(dataset_name: str,
                                clinical: pd.DataFrame,
                                transcripts: dict[str, dict[str, int]]
                                ) -> pd.DataFrame:
    """Return one row per transcript with word count < MIN_WORD_COUNT, annotated
    with caliber_code, task, word_count, and symptom_state."""
    clinical = clinical.copy()
    clinical["caliber_code"]  = clinical["caliber_code"].astype(str).str.strip().str.lower()
    clinical["symptom_state"] = _fold_mixed(clinical["symptom_state"])
    state_lookup = clinical.set_index("caliber_code")["symptom_state"].to_dict()

    rows: list[dict] = []
    for task, code_to_wc in transcripts.items():
        for code, wc in code_to_wc.items():
            if wc < MIN_WORD_COUNT:
                rows.append({
                    "dataset":       dataset_name,
                    "caliber_code":  code,
                    "task":          task,
                    "word_count":    wc,
                    "symptom_state": state_lookup.get(code, "unknown"),
                    "in_clinical":   code in state_lookup,
                })
    return pd.DataFrame(rows)


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_availability(df: pd.DataFrame) -> None:
    """Print the transcript availability table to the console."""
    if df.empty:
        print("  ⚠ Availability table is empty.")
        return
    print(f"\n{'═'*70}\n  Transcript availability (threshold: ≥{MIN_WORD_COUNT} words)\n{'═'*70}")
    print(f"  {'dataset':<14}{'task':<24}{'available':>14}{'%':>8}")
    print(f"  {'-'*14}{'-'*24}{'-'*14}{'-'*8}")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:<14}{r['task']:<24}"
              f"{int(r['n_available']):>6}/{int(r['n_total']):<6}"
              f"{r['pct_available']:>7.1f}")


def _print_descriptives(df: pd.DataFrame) -> None:
    """Print the word-count descriptives table to the console."""
    if df.empty:
        return
    print(f"\n{'═'*70}\n  Word-count descriptives (all found transcripts)\n{'═'*70}")
    print(f"  {'dataset':<14}{'task':<24}{'n':>5}{'<10':>6}"
          f"{'min':>6}{'max':>6}{'med':>8}{'mean':>8}{'sd':>8}")
    print(f"  {'-'*14}{'-'*24}{'-'*5}{'-'*6}{'-'*6}{'-'*6}{'-'*8}{'-'*8}{'-'*8}")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:<14}{r['task']:<24}"
              f"{int(r['n_found']):>5}{int(r['n_below_threshold']):>6}"
              f"{int(r['min']):>6}{int(r['max']):>6}"
              f"{r['median']:>8.1f}{r['mean']:>8.1f}{r['sd']:>8.1f}")


def _print_exclusion_candidates(df: pd.DataFrame) -> None:
    """Print the list of transcripts below the word-count threshold."""
    if df.empty:
        print("\n  ✓ No transcripts below the word-count threshold.")
        return
    print(f"\n{'═'*70}\n  Exclusion candidates (word count < {MIN_WORD_COUNT})\n{'═'*70}")
    print(f"  {'dataset':<14}{'caliber_code':<18}{'task':<24}"
          f"{'wc':>5}{'state':<14}{'in_clinical':>12}")
    print(f"  {'-'*14}{'-'*18}{'-'*24}{'-'*5}{'-'*14}{'-'*12}")
    for _, r in df.sort_values(["dataset", "task", "word_count"]).iterrows():
        print(f"  {r['dataset']:<14}{r['caliber_code']:<18}{r['task']:<24}"
              f"{int(r['word_count']):>5}  {r['symptom_state']:<12}"
              f"{'yes' if r['in_clinical'] else 'NO':>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_transcript_analysis() -> None:
    """Entry point: index the cleaned-monologue folders, then build and save the
    availability, word-count descriptive, and exclusion-candidate tables for the
    unique and longitudinal datasets."""
    print("═" * 70)
    print("  TRANSCRIPT ANALYSIS  (data-section descriptor)")
    print("═" * 70)

    unique_df = pd.read_csv(AUDIO_DATA_UNIQUE)
    long_df   = pd.read_csv(AUDIO_DATA_LONGITUDINAL)
    
    unique_codes = set(unique_df["caliber_code"].astype(str).str.strip().str.lower())
    long_codes   = set(long_df["caliber_code"].astype(str).str.strip().str.lower())

    print(f"\n  Unique-patient sessions:  {len(unique_df)}")
    print(f"  Longitudinal sessions:    {len(long_df)}")

    # Index every task folder once, then split by dataset.
    print("\n  Indexing cleaned-monologue folders…")
    transcripts_unique: dict[str, dict[str, int]] = {}
    transcripts_long:   dict[str, dict[str, int]] = {}

    for task, folder in TASK_DIRS.items():
        all_codes = _index_task_folder(task, folder)
        transcripts_unique[task] = {c: wc for c, wc in all_codes.items()
                                    if c in unique_codes}
        transcripts_long[task]   = {c: wc for c, wc in all_codes.items()
                                    if c in long_codes}
        print(f"    {task:<24}"
              f"unique={len(transcripts_unique[task])}  "
              f"longitudinal={len(transcripts_long[task])}")

    # ── 1. Availability ───────────────────────────────────────────────────────
    rows_u, warns_u = _build_availability("unique",       unique_df, transcripts_unique)
    rows_l, warns_l = _build_availability("longitudinal", long_df,   transcripts_long)
    availability = pd.concat([rows_u, rows_l], ignore_index=True)
    _print_availability(availability)

    if warns_u or warns_l:
        print(f"\n  ⚠ State-asymmetric dropout flagged (spread > {DROPOUT_FLAG_PP:.0f} pp):")
        for w in warns_u + warns_l:
            print(f"    {w}")
    else:
        print(f"\n  ✓ No state-asymmetric dropout above {DROPOUT_FLAG_PP:.0f} pp.")

    # ── 2. Word-count descriptives ────────────────────────────────────────────
    desc_u = _build_descriptives("unique",       unique_df, transcripts_unique)
    desc_l = _build_descriptives("longitudinal", long_df,   transcripts_long)
    descriptives = pd.concat([desc_u, desc_l], ignore_index=True)
    _print_descriptives(descriptives)

    # ── 3. Exclusion candidates ───────────────────────────────────────────────
    excl_u = _build_exclusion_candidates("unique",       unique_df, transcripts_unique)
    excl_l = _build_exclusion_candidates("longitudinal", long_df,   transcripts_long)
    exclusions = pd.concat([excl_u, excl_l], ignore_index=True)
    _print_exclusion_candidates(exclusions)

    # ── Save outputs ──────────────────────────────────────────────────────────
    availability.to_csv(RESULTS_SUBDIR / "transcript_availability.csv",    index=False)
    descriptives.to_csv(RESULTS_SUBDIR / "word_count_descriptives.csv",    index=False)
    exclusions.to_csv(  RESULTS_SUBDIR / "exclusion_candidates.csv",       index=False)

    print(f"\n  ✓ Results saved to {RESULTS_SUBDIR}")
    print("\n  Done.")


if __name__ == "__main__":
    run_transcript_analysis()