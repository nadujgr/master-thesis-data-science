"""
Cross-methodology check that every patient-level CSV carries an `experiment`
column, uses the same experiment naming as its matching stats CSV, has no
duplicate (caliber_code, experiment) rows, and has the standard metadata columns.
Run after re-executing the feature scripts: downstream steps join on
`experiment`, so mismatches here surface as missing or empty results. Returns a
nonzero exit code if any methodology fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR

METHODOLOGIES = [
    ("speech_graphs",
     "speech_graphs/speech_graphs_patient_values.csv",
     "speech_graphs/speech_graphs_per_experiment.csv"),
    ("semantic_networks",
     "semantic_networks/semantic_networks_patient_features.csv",
     "semantic_networks/semantic_networks_per_experiment.csv"),
    ("semantic_similarity",
     "semantic_similarity/semantic_patient_features.csv",
     "semantic_similarity/semantic_per_experiment.csv"),
    ("cooccurrence_emotion",
     "cooccurrence_emotion/cooccurrence_emotion_patient_values.csv",
     "cooccurrence_emotion/cooccurrence_emotion_long.csv"),
    ("cooccurrence",
     "cooccurrence/cooccurrence_patient_values.csv",
     "cooccurrence/cooccurrence_long.csv"),
    ("lexical",
     "lexical/lexical_patient_features.csv",
     "lexical/lexical_per_task.csv"),
    ("psycholinguistic",
     "psycholinguistic/psycholinguistic_patient_features.csv",
     "psycholinguistic/psycholinguistic_per_task.csv"),
    ("topic_analysis",
     "topic_analysis/minilm/topics_shape_patient_values.csv",
     "topic_analysis/minilm/topics_shape_per_experiment.csv"),
]

REQUIRED_PATIENT_COLS = {"caliber_code"}


def _check(name: str, patient_path: Path, stats_path: Path) -> bool:
    """Verify one methodology's patient CSV against its stats CSV: confirm the
    required metadata and `experiment` columns exist, that the experiment naming
    matches the stats file, and that there are no duplicate
    (caliber_code, experiment) rows. Prints a per-check report and returns True
    when all fatal checks pass."""
    print(f"\n{name}")
    print("─" * 78)

    if not patient_path.exists():
        print(f"  ✗ Patient CSV missing: {patient_path}")
        return False
    if not stats_path.exists():
        print(f"  ⚠ Stats CSV missing:  {stats_path}")
        return True   # not fatal — patient file alone can't be verified

    pat = pd.read_csv(patient_path)
    sta = pd.read_csv(stats_path)
    ok  = True

    # Required columns
    missing = REQUIRED_PATIENT_COLS - set(pat.columns)
    if missing:
        print(f"  ✗ Patient CSV missing columns: {missing}")
        ok = False
    # accept either symptom_state or episode_type (cooccurrence uses the latter)
    if not ({"symptom_state", "episode_type"} & set(pat.columns)):
        print(f"  ✗ Patient CSV missing symptom_state / episode_type")
        ok = False

    if "experiment" not in pat.columns:
        print(f"  ✗ Patient CSV missing 'experiment' column")
        print(f"      (re-run the corresponding analysis script)")
        return False
    print(f"  ✓ Patient CSV has 'experiment' column")

    # Experiment value sets
    pat_exp = set(pat["experiment"].dropna().unique())
    if "experiment" in sta.columns:
        sta_exp = set(sta["experiment"].dropna().unique())
        missing_in_pat = sta_exp - pat_exp
        extra_in_pat   = pat_exp - sta_exp
        if missing_in_pat:
            print(f"  ✗ Stats experiments missing from patient CSV:")
            for e in sorted(missing_in_pat)[:10]:
                print(f"      {e}")
            ok = False
        if extra_in_pat:
            print(f"  ⚠ Patient experiments not present in stats CSV:")
            for e in sorted(extra_in_pat)[:10]:
                print(f"      {e}")
        if not missing_in_pat:
            print(f"  ✓ All {len(sta_exp)} stats experiments present in patient CSV")
    else:
        print(f"  ⚠ Stats CSV has no 'experiment' column — cannot verify naming match")
        print(f"      Patient experiments seen: {sorted(pat_exp)[:5]}...")

    # Duplicate detection
    dups = pat.groupby(["caliber_code", "experiment"], dropna=False).size()
    n_dup = int((dups > 1).sum())
    if n_dup:
        print(f"  ✗ {n_dup} duplicate (caliber_code, experiment) pairs found")
        ok = False
    else:
        print(f"  ✓ No duplicates: each patient appears once per experiment")

    print(f"      {len(pat)} rows  |  {pat['caliber_code'].nunique()} patients  "
          f"|  {len(pat_exp)} experiments")
    return ok


def verify_patient_data_consistency() -> int:
    """Run the consistency check across every configured methodology and print an
    overall pass/fail summary. Returns 0 if all methodologies pass, 1 otherwise
    (suitable as a process exit code)."""
    print("═" * 78)
    print(" CROSS-METHODOLOGY PATIENT-DATA CONSISTENCY CHECK")
    print("═" * 78)

    all_ok = True
    for name, patient_rel, stats_rel in METHODOLOGIES:
        patient_path = Path(RESULTS_DIR) / patient_rel
        stats_path   = Path(RESULTS_DIR) / stats_rel
        if not _check(name, patient_path, stats_path):
            all_ok = False

    print("\n" + "═" * 78)
    if all_ok:
        print(" ✓ ALL CHECKS PASSED")
    else:
        print(" ✗ ONE OR MORE METHODOLOGIES FAILED — see details above")
    print("═" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(verify_patient_data_consistency())
