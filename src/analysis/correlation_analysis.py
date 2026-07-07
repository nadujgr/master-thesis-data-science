"""
Within-family feature redundancy (diagnostic only).

Computes the within-family Spearman correlation structure for the four retained
feature families on the a-priori fixed emotion-task pipeline (speech graphs
emotion_w20s10, cooccurrence emotion_ws3, lexical and psycholinguistic). This is
label-blind and does not select or remove features; it only reports redundancy
so a threshold can be inspected before use in the classification pipeline.
A-priori redundant blocks (listed in APRIORI_DROP) are dropped by name before
correlation because they duplicate retained features. Per family it writes a
signed correlation matrix, a redundant-pairs list (|r| >= REPORT_MIN), and a
per-feature summary into OUTPUT_DIR/correlation/<family>/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, CLASSIFICATION_RESULTS_DIR
try:
    from src.stats_helpers import fold_mixed
except Exception:
    # Fall back so the diagnostic still runs if the helper import path differs.
    def fold_mixed(df):
        return df


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

_RES        = Path(RESULTS_DIR)
OUTPUT_DIR  = Path(CLASSIFICATION_RESULTS_DIR) / "correlation"

TASK_VALUE  = "emotion"

# |r| at/above which a pair is reported as redundant
REPORT_MIN  = 0.70
THRESHOLD_BANDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NZV_MIN_STD = 1e-8   # flag (not drop) columns flatter than this

# Column-name handling
COLS = {
    "id":         "caliber_code",
    "state":      "symptom_state",
    "state_alt":  "episode_type",
    "task":       "task",
    "task_alt":   "task_set",
    "experiment": "experiment",
    "word_count": "word_count",
}

# Metadata / non-feature columns (lower-cased compare). word_count is excluded
# as a covariate/baseline, analysed separately.
_METADATA = {
    "caliber_code", "symptom_state", "episode_type", "visit", "task",
    "task_set", "window", "experiment", "model", "ymrs", "hdrs", "sex",
    "age", "language", "lang", "word_count", "subject", "patient_id", "id",
    "group", "diagnosis",
}

# Per-family source files and the pinned experiment to filter to (or None).
FAMILIES = {
    "lexical": {
        "path":       _RES / "lexical" / "lexical_patient_features.csv",
        "experiment": None,
    },
    "psycholinguistic": {
        "path":       _RES / "psycholinguistic" / "psycholinguistic_patient_features.csv",
        "experiment": None,
    },
    "speech_graphs": {
        "path":       _RES / "speech_graphs" / "speech_graphs_patient_values.csv",
        "experiment": "emotion_w20s10",
    },
    "cooccurrence_emotion": {
        "path":       _RES / "cooccurrence_emotion" / "cooccurrence_emotion_patient_values.csv",
        "experiment": "emotion_ws3",
    },
}

# A-priori redundant blocks, dropped by name before correlation because they
# duplicate retained features (not because of missingness).
APRIORI_DROP = {
    "lexical": [],
    "psycholinguistic": [],
    "speech_graphs": [
        "LCC",          # equals Nodes on every row
    ],
    "cooccurrence_emotion": [
        # z-standardised variants of retained raw features (undefined for
        # one-sided / short transcripts)
        "valence_homophily_z", "arousal_homophily_z", "dominance_homophily_z",
        "valence_dob_z", "arousal_dob_z", "dominance_dob_z",
        "emotional_complexity_z_shuffled", "emotional_complexity_z_uniform",
        # pagerank block: monotonic with degree on undirected graphs, redundant
        # with the retained degree/eigenvector/closeness centralities
        "mean_pagerank", "std_pagerank", "skew_pagerank",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, renaming the legacy 'task_set' and 'episode_type'
    columns to the canonical 'task' and 'symptom_state' when those are absent."""
    if COLS["task"] not in df.columns and COLS["task_alt"] in df.columns:
        df = df.rename(columns={COLS["task_alt"]: COLS["task"]})
    if COLS["state"] not in df.columns and COLS["state_alt"] in df.columns:
        df = df.rename(columns={COLS["state_alt"]: COLS["state"]})
    return df


def _feature_columns(df: pd.DataFrame, dropped: list[str]) -> list[str]:
    """Return the numeric feature columns for correlation, excluding metadata /
    covariate columns and any names listed in the a-priori drop list
    (case-insensitive)."""
    numeric = df.select_dtypes(include=[np.number]).columns
    drop_lc = {d.lower() for d in dropped}
    feats = [c for c in numeric
             if c.lower() not in _METADATA and c.lower() not in drop_lc]
    return feats


def _nan_map(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Which patient × which feature is NaN (for the console diagnostic)."""
    sub = df[[COLS["id"]] + feats]
    rows = []
    for _, r in sub.iterrows():
        miss = [f for f in feats if pd.isna(r[f])]
        if miss:
            rows.append({"caliber_code": r[COLS["id"]],
                         "n_missing": len(miss),
                         "missing_features": ", ".join(miss)})
    return pd.DataFrame(rows)


def _redundant_pairs(corr_abs: pd.DataFrame, feats: list[str],
                     report_min: float) -> pd.DataFrame:
    """Long list of feature pairs with |r| >= report_min, sorted desc."""
    out = []
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1:]:
            r = corr_abs.loc[f1, f2]
            if pd.notna(r) and r >= report_min:
                # Feature an auto-rule would drop: higher mean |r| to the rest
                m1 = corr_abs[f1].drop(index=f1).mean()
                m2 = corr_abs[f2].drop(index=f2).mean()
                would_drop = f1 if m1 > m2 else f2
                out.append({
                    "feature_1": f1, "feature_2": f2,
                    "abs_r": round(float(r), 4),
                    "mean_absr_f1": round(float(m1), 4),
                    "mean_absr_f2": round(float(m2), 4),
                    "auto_rule_would_drop": would_drop,
                })
    return (pd.DataFrame(out).sort_values("abs_r", ascending=False)
            .reset_index(drop=True) if out else pd.DataFrame(
                columns=["feature_1", "feature_2", "abs_r",
                         "mean_absr_f1", "mean_absr_f2", "auto_rule_would_drop"]))


def _band_counts(corr_abs: pd.DataFrame, feats: list[str]) -> dict:
    """How many redundant pairs exist at each threshold band."""
    counts = {}
    triu = []
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1:]:
            r = corr_abs.loc[f1, f2]
            if pd.notna(r):
                triu.append(r)
    arr = np.array(triu)
    for t in THRESHOLD_BANDS:
        counts[t] = int((arr >= t).sum())
    return counts


# ═══════════════════════════════════════════════════════════════════
# PER-FAMILY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyse_family(name: str, cfg: dict) -> dict | None:
    """Compute the within-family Spearman redundancy structure for one feature
    family. Loads and filters the family's CSV to the emotion task and pinned
    experiment, folds mixed episodes, deduplicates patients, drops the a-priori
    redundant blocks by name, prints console diagnostics (NaN map, near-zero
    variance, complete-case count), and returns the signed/absolute correlation
    matrices, redundant-pair list, per-feature summary and threshold-band counts.
    Returns None when the file is missing or there is too little data."""
    path = cfg["path"]
    if not Path(path).exists():
        print(f"\n[{name}] ⚠ file not found: {path}")
        return None

    df = _norm_cols(pd.read_csv(path))

    # Filter to emotion task (if present) and the pinned experiment
    if COLS["task"] in df.columns:
        df = df[df[COLS["task"]].astype(str) == TASK_VALUE]
    if cfg["experiment"] is not None and COLS["experiment"] in df.columns:
        df = df[df[COLS["experiment"]].astype(str) == cfg["experiment"]]

    if COLS["id"] not in df.columns:
        print(f"\n[{name}] ⚠ no '{COLS['id']}' column — skip")
        return None

    df[COLS["id"]] = df[COLS["id"]].astype(str).str.strip().str.lower()
    df = fold_mixed(df)
    df = df.drop_duplicates(COLS["id"]).reset_index(drop=True)

    dropped = APRIORI_DROP.get(name, [])
    present_drops = [d for d in dropped if d in df.columns]
    feats = _feature_columns(df, dropped)

    # ── Console diagnostics ──
    print("\n" + "═" * 68)
    print(f"[{name}]  source: {Path(path).name}")
    if cfg["experiment"]:
        print(f"   experiment filter: {cfg['experiment']}")
    print(f"   patients (after dedup): {len(df)}")
    print(f"   a-priori dropped (redundant): "
          f"{present_drops if present_drops else '—'}")
    print(f"   features entering correlation: {len(feats)}")

    # NaN map after the a-priori drops
    nan_df = _nan_map(df, feats)
    if not nan_df.empty:
        print(f"   ⚠ remaining NaNs in {len(nan_df)} patient(s):")
        for _, r in nan_df.iterrows():
            print(f"       {r['caliber_code']}: {r['missing_features']}")
    else:
        print(f"   no NaNs remain in feature columns ✓")

    # Near-zero-variance flag (report, don't drop)
    stds = df[feats].std(skipna=True)
    flat = stds[stds <= NZV_MIN_STD].index.tolist()
    if flat:
        print(f"   ⚠ near-zero-variance (flagged, not dropped): {flat}")

    # Complete-case patient set for a consistent matrix
    complete = df.loc[df[feats].notna().all(axis=1)].reset_index(drop=True)
    n_complete = len(complete)
    print(f"   complete-case patients for correlation: {n_complete}")

    if len(feats) < 2 or n_complete < 5:
        print(f"   ⚠ not enough features/patients for correlation — skip")
        return None

    # ── Spearman correlation (signed + abs) ──
    corr_signed = complete[feats].corr(method="spearman")
    corr_abs    = corr_signed.abs()

    pairs_df = _redundant_pairs(corr_abs, feats, REPORT_MIN)
    bands    = _band_counts(corr_abs, feats)

    print(f"   redundant pairs by threshold: "
          + "  ".join(f"|r|≥{t}:{bands[t]}" for t in THRESHOLD_BANDS))

    # per-feature summary
    summary = pd.DataFrame({
        "feature":        feats,
        "n":              [int(complete[f].notna().sum()) for f in feats],
        "std":            [round(float(complete[f].std()), 6) for f in feats],
        "mean_abs_r_to_others": [
            round(float(corr_abs[f].drop(index=f).mean()), 4) for f in feats],
        "max_abs_r_to_others": [
            round(float(corr_abs[f].drop(index=f).max()), 4) for f in feats],
        "near_zero_variance": [f in flat for f in feats],
        "a_priori_dropped":   False,
    })
    # append the dropped blocks for visibility
    if present_drops:
        summary = pd.concat([summary, pd.DataFrame({
            "feature": present_drops, "n": np.nan, "std": np.nan,
            "mean_abs_r_to_others": np.nan, "max_abs_r_to_others": np.nan,
            "near_zero_variance": np.nan, "a_priori_dropped": True,
        })], ignore_index=True)

    return {
        "name":        name,
        "n_patients":  len(df),
        "n_complete":  n_complete,
        "features":    feats,
        "corr_signed": corr_signed,
        "corr_abs":    corr_abs,
        "pairs":       pairs_df,
        "summary":     summary,
        "bands":       bands,
        "dropped":     present_drops,
        "nan_map":     nan_df,
    }


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

def write_family_csvs(res: dict) -> None:
    """Write the three per-family diagnostic CSVs (signed correlation matrix,
    redundant-pair list, and per-feature summary) into the family's output
    directory."""
    fam_dir = OUTPUT_DIR / res["name"]
    fam_dir.mkdir(parents=True, exist_ok=True)

    # full matrix (signed r; abs is easily derived). Keep index for readability.
    res["corr_signed"].round(4).to_csv(
        fam_dir / f"{res['name']}_correlation_matrix.csv")
    res["pairs"].to_csv(
        fam_dir / f"{res['name']}_redundant_pairs.csv", index=False)
    res["summary"].to_csv(
        fam_dir / f"{res['name']}_feature_summary.csv", index=False)

    print(f"   → {fam_dir}/")
    print(f"        {res['name']}_correlation_matrix.csv")
    print(f"        {res['name']}_redundant_pairs.csv")
    print(f"        {res['name']}_feature_summary.csv")


def write_overview(results: list[dict]) -> None:
    """Write and print the cross-family overview table summarising, per family,
    the patient and feature counts and the number of redundant pairs at each
    correlation threshold band."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = {
            "family":      r["name"],
            "n_patients":  r["n_patients"],
            "n_complete":  r["n_complete"],
            "n_features":  len(r["features"]),
            "n_dropped_apriori": len(r["dropped"]),
        }
        for t in THRESHOLD_BANDS:
            row[f"pairs_ge_{t}"] = r["bands"][t]
        rows.append(row)
    ov = pd.DataFrame(rows)
    ov.to_csv(OUTPUT_DIR / "correlation_overview.csv", index=False)
    print("\n" + "═" * 68)
    print("OVERVIEW (redundant-pair counts by threshold, per family)")
    print("═" * 68)
    print(ov.to_string(index=False))
    print(f"\n   → {OUTPUT_DIR / 'correlation_overview.csv'}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_correlation():
    """Run the within-family correlation diagnostic across all configured
    families: analyse each, write its per-family CSVs, and write the overall
    overview. Reports only — no features are selected or removed."""
    print("═" * 68)
    print("WITHIN-FAMILY CORRELATION ANALYSIS  (diagnostic only)")
    print(f"task={TASK_VALUE} | Spearman | report |r|≥{REPORT_MIN} | label-blind")
    print("═" * 68)

    results = []
    for name, cfg in FAMILIES.items():
        res = analyse_family(name, cfg)
        if res is not None:
            write_family_csvs(res)
            results.append(res)

    if results:
        write_overview(results)
    print("\nDone. No features were removed — inspect the CSVs, then set the "
          "applied threshold in the classification pipeline.")


if __name__ == "__main__":
    run_correlation()