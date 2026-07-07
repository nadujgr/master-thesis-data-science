"""
Speech graph analysis runner.

Loops over a set of experiments (task x window/step), runs the shared analysis on
each, and writes speech_graphs_per_experiment.csv (one row per experiment x
metric) and speech_graphs_consistency.csv (one row per metric across
experiments). Optionally also runs the parallel z-score (structure-vs-random)
analysis, producing two more CSVs with a '_z' suffix. 'mixed' is folded into
'mania', and word_count and language are adjusted for inside the rank-based
ANCOVA.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (AUDIO_DATA_UNIQUE, SPEECHGRAPH_DATA, AUDIO_DATA_LONGITUDINAL,
                        TRANSCRIPT_BY_PART_MONOLOGUE, RESULTS_DIR)

from src.stats_helpers import (analyze_experiment, consistency_across_experiments,
                      fold_mixed, print_experiment_summary,
                      print_consistency_summary, STATES)
from src.utils import clean_caliber_code_series as _clean_code

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_NAME = "speech_graphs"
OUTPUT_DIR = Path(RESULTS_DIR) / LEVEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["LCC", "LSC", "Nodes", "Edges", "ATD", "Density",
           "ASP", "Diameter", "CC", "RE", "PE", "L1", "L2", "L3"]

TASK_DIRS = {
    "emotion": {
        "speechgraph": os.path.join(SPEECHGRAPH_DATA,            "emotion_clean"),
        "monologue":   os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "emotion_clean"),
    },
    "bdae": {
        "speechgraph": os.path.join(SPEECHGRAPH_DATA,            "bdae_clean"),
        "monologue":   os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "bdae_clean"),
    },
    "clinical_evaluation": {
        "speechgraph": os.path.join(SPEECHGRAPH_DATA,            "clinical_evaluation_clean"),
        "monologue":   os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "clinical_evaluation_clean"),
    },
}

EXPERIMENTS = [
    {"name": "emotion_w10s1",  "task": "emotion",             "folder": "results_split_10_1_100rand"},
    {"name": "bdae_w10s1",     "task": "bdae",                "folder": "results_split_10_1_100rand"},
    {"name": "clinical_w10s1", "task": "clinical_evaluation", "folder": "results_split_10_1_100rand"},
    # different window/step sizes
    {"name": "emotion_w30s15",  "task": "emotion",             "folder": "results_split_30_15_100rand"},
    {"name": "bdae_w30s15",     "task": "bdae",                "folder": "results_split_30_15_100rand"},
    {"name": "clinical_w30s15", "task": "clinical_evaluation", "folder": "results_split_30_15_100rand"},
    {"name": "emotion_w20s10",  "task": "emotion",             "folder": "results_split_20_10_100rand"},
    {"name": "bdae_w20s10",     "task": "bdae",                "folder": "results_split_20_10_100rand"},
    {"name": "clinical_w20s10", "task": "clinical_evaluation", "folder": "results_split_20_10_100rand"},
    {"name": "emotion_w10s5",   "task": "emotion",             "folder": "results_split_10_5_100rand"},
    {"name": "bdae_w10s5",      "task": "bdae",                "folder": "results_split_10_5_100rand"},
    {"name": "clinical_w10s5",  "task": "clinical_evaluation", "folder": "results_split_10_5_100rand"},

]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _save(rows: list, filename: str, out_dir: Path) -> None:
       """Concatenate a list of dataframes (or dicts) and save to CSV."""
       if not rows:
           return
       if isinstance(rows[0], pd.DataFrame):
           df = pd.concat(rows, ignore_index=True)
       else:
           df = pd.DataFrame(rows)
       df.to_csv(out_dir / filename, index=False)


def _word_counts(monologue_dir: str, task: str) -> pd.DataFrame:
    """Count words in every transcript file for one task, returning a
    caliber_code / word_count dataframe. Files are read as windows-1252, falling
    back to utf-8; the caliber code is recovered from the file name. Returns an
    empty frame if the directory does not exist."""
    counts = {}
    if not os.path.isdir(monologue_dir):
        return pd.DataFrame(columns=["caliber_code", "word_count"])
    for fname in os.listdir(monologue_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(monologue_dir, fname)
        try:
            with open(path, encoding="windows-1252") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        code = (fname.replace(f"_{task}.txt", "")
                     .replace(".txt", "").strip().lower())
        counts[code] = len(text.split())
    return pd.DataFrame(counts.items(), columns=["caliber_code", "word_count"])


def load_experiment(exp: dict, df_unique: pd.DataFrame
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Load one experiment's metrics, clinical info and word counts, returning
    (df, available_metric_columns). Adds '_z' columns if zscore.txt is present so
    the parallel z-score analysis can reuse the same df."""
    dirs = TASK_DIRS[exp["task"]]
    results_dir   = os.path.join(dirs["speechgraph"], exp["folder"])
    monologue_dir = dirs["monologue"]

    # Per-window mean metrics (windowed graphs)
    df_real = pd.read_csv(os.path.join(results_dir, "means_params_table.txt"))
    df_real = df_real[~df_real["File"].str.contains("_random")].copy()
    df_real["caliber_code"] = _clean_code(df_real["File"])

    df_wc = _word_counts(monologue_dir, exp["task"])

    clinical_cols = [c for c in ["caliber_code", "symptom_state", "language",
                                  "sex", "age", "ymrs", "hdrs", "visit"]
                     if c in df_unique.columns]

    df = (df_real
          .merge(df_unique[clinical_cols], on="caliber_code", how="inner")
          .merge(df_wc, on="caliber_code", how="left")
          .copy())
    df = fold_mixed(df)

    # Drop patients below the window threshold (all metrics NaN)
    metric_present = [c for c in METRICS if c in df.columns]
    n0 = len(df)
    df = df.dropna(subset=metric_present, how="all").copy()
    if len(df) < n0:
        print(f"  Dropped {n0 - len(df)} patients (below window threshold)")

    # Z-scores (structure vs word-shuffled baseline) — added if available
    z_path = os.path.join(results_dir, "zscore.txt")
    if os.path.exists(z_path):
        dz = pd.read_csv(z_path)
        dz["caliber_code"] = _clean_code(dz["File"])
        dz = dz[dz["File"].str.contains("_random")]
        z_cols = [m for m in METRICS if m in dz.columns]
        dz = (dz[["caliber_code"] + z_cols]
              .rename(columns={m: f"{m}_z" for m in z_cols}))
        df = df.merge(dz, on="caliber_code", how="left")

    available = [m for m in METRICS if m in df.columns and df[m].nunique() > 1]

    counts = df["symptom_state"].value_counts()
    print(f"  n={len(df)}  "
          + " ".join(f"{s}={counts.get(s, 0)}" for s in STATES))
    return df, available


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_speech_graph_analysis(
    metric_suffix: str = "",
    longitudinal: bool = False,
) -> None:
    """Run the speech-graph analysis across all experiments (task x window/step):
    load windowed graph metrics, clinical labels and word counts, save the
    per-patient (or per-session) values, and, in cross-sectional mode, run the
    per-experiment analysis and cross-experiment consistency summary. Passing
    metric_suffix='_z' runs the structure-vs-random z-score variant; longitudinal
    mode saves session values and defers statistics to the longitudinal script."""

    label = "Z-SCORE" if metric_suffix == "_z" else "RAW"
    mode  = "LONGITUDINAL" if longitudinal else "CROSS-SECTIONAL"
    print("═" * 70)
    print(f"  SPEECH GRAPH ANALYSIS — {label} ({mode})")
    print("═" * 70)

    clinical_path = AUDIO_DATA_LONGITUDINAL if longitudinal else AUDIO_DATA_UNIQUE
    output_dir    = Path(RESULTS_DIR) / ("longitudinal/speech_graphs" if longitudinal
                                          else "speech_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_clinical = pd.read_csv(clinical_path)

    all_long = []
    all_patient_rows = []

    for exp in EXPERIMENTS:
        print(f"\n── {exp['name']} ──")
        try:
            df, base_metrics = load_experiment(exp, df_clinical)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue

        # One row per (patient x experiment), not averaged across windows.
        # `experiment` matches speech_graphs_per_experiment.csv for joining.
        if metric_suffix == "":
                keep = (["caliber_code", "symptom_state"] +
                        (["visit"] if "visit" in df.columns else []) +
                        [c for c in ["language", "word_count", "ymrs", "hdrs", "sex", "age"]
                        if c in df.columns] +
                        [m for m in METRICS if m in df.columns] +
                        [c for c in df.columns if c.endswith("_z")])
                pat               = df[keep].copy()
                pat["task"]       = exp["task"]
                pat["window"]     = exp["folder"]
                pat["experiment"] = exp["name"]
                all_patient_rows.append(pat)

        if metric_suffix == "_z":
            metric_cols = [f"{m}_z" for m in base_metrics
                           if f"{m}_z" in df.columns]
            if not metric_cols:
                print("  ✗ No z-score columns — skipping.")
                continue
        else:
            metric_cols = base_metrics

        result = analyze_experiment(df, metric_cols)
        if result.empty:
            continue

        result.insert(0, "experiment", exp["name"])
        result.insert(1, "task",       exp["task"])
        result.insert(2, "window",     exp["folder"])
        result.insert(3, "is_primary", bool(exp.get("primary", False)))
        all_long.append(result)
        print_experiment_summary(exp["name"], result)

    if metric_suffix == "" and all_patient_rows:
        pat_df = pd.concat(all_patient_rows, ignore_index=True)

        if longitudinal:
            out_pat = output_dir / "speech_graphs_sessions.csv"
            pat_df.to_csv(out_pat, index=False)
            print(f"  ✓ Sessions saved → {out_pat}")
            print(f"    {pat_df['caliber_code'].nunique()} patients, "
                  f"{len(pat_df)} session-experiment observations")
            print("  longitudinal=True — skipping cross-sectional analysis.")
            return

        out_pat = output_dir / "speech_graphs_patient_values.csv"
        pat_df.to_csv(out_pat, index=False)
        print(f"  ✓ Saved {len(pat_df)} patient-experiment observations → {out_pat}")

    if longitudinal:
        return

    if not all_long:
        print("\n✗ No experiments produced results.")
        return

    long_df = pd.concat(all_long, ignore_index=True)
    summary = consistency_across_experiments(long_df)

    out_long    = output_dir / f"speech_graphs_per_experiment{metric_suffix}.csv"
    out_summary = output_dir / f"speech_graphs_consistency{metric_suffix}.csv"
    long_df.to_csv(out_long,    index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\n{'═'*70}\n  CONSISTENCY ACROSS EXPERIMENTS ({label})\n{'═'*70}")
    print_consistency_summary(summary)
    print(f"\n  Wrote: {out_long}")
    print(f"         {out_summary}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--longitudinal", action="store_true", default=False)
    parser.add_argument("--zscore",       action="store_true", default=False)
    args = parser.parse_args()

    run_speech_graph_analysis(
        metric_suffix="_z" if args.zscore else "",
        longitudinal=args.longitudinal,
    )