"""
Builds the merged longitudinal dataset. Extracts per-methodology session
features (only for families whose cached *_sessions.csv is missing), merges them
on patient/visit/task, joins clinical metadata, and writes long_features.csv.
Set FORCE_EXTRACT=True to re-extract everything. The analysis is a separate
script (longitudinal_analysis.py) that reads the CSV produced here.
"""

import importlib
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, AUDIO_DATA_LONGITUDINAL


LONG_RESULTS = Path(RESULTS_DIR) / "longitudinal"
CLINICAL_CSV = AUDIO_DATA_LONGITUDINAL
OUT_CSV      = LONG_RESULTS / "long_features.csv"
FORCE_EXTRACT = False   # re-run extraction even if cached CSVs exist

# Each family maps to the cached CSV it writes and the (module, function)
# extractor that produces it. "features": None means auto-detect numeric columns.
METHODOLOGIES = {
    "lexical": {
        "csv": LONG_RESULTS / "lexical/lexical_sessions.csv",
        "extractor": ("src.features.lexical", "run_lexical"),
        "features": ["word_count", "type_token_ratio", "hapax_ratio",
                     "mean_word_length", "long_word_ratio", "short_word_ratio",
                     "repetition_rate"],
    },
    "psycholinguistic": {
        "csv": LONG_RESULTS / "psycholinguistic/psycholinguistic_sessions.csv",
        "extractor": ("src.features.psycholinguistics", "run_psycholinguistic"),
        "features": None,
    },
    "speech_graphs": {
        "csv": LONG_RESULTS / "speech_graphs/speech_graphs_sessions.csv",
        "extractor": ("src.features.speech_graphs", "run_speech_graph_analysis"),
        "features": ["LCC", "LSC", "ATD", "Density", "ASP", "Diameter",
                     "CC", "RE", "PE", "L1", "L2", "L3"],
    },
    "cooccurrence_emotion": {
        "csv": LONG_RESULTS / "cooccurrence_emotion/cooccurrence_emotion_sessions.csv",
        "extractor": ("src.features.cooccurrence_emotion", "run_cooccurrence_nrc_analysis"),
        "features": None,
    },
}

# Handling of multiple elicitation tasks per visit:
#   None     -> keep a `task` column (one row per patient x visit x task).
#   "mean"   -> average features across tasks (one row per patient x visit).
#   "<name>" -> keep only that single task (e.g. "bdae").
COLLAPSE_TASKS = None

_META = {"caliber_code", "patient_id", "id", "task", "experiment", "window",
         "visit", "symptom_state", "mood_state", "ymrs", "hdrs", "language",
         "sex", "age", "n_windows", "oov_rate"}


def ensure_extracted(force: bool = False) -> None:
    """Run feature extraction for each family whose cached CSV is absent.
    Extractors are imported lazily so this file runs standalone when the CSVs
    already exist."""
    for name, cfg in METHODOLOGIES.items():
        csv = Path(cfg["csv"])
        if csv.exists() and not force:
            print(f"  ✓ {name}: cached CSV found — skipping extraction")
            continue
        mod_name, func_name = cfg["extractor"]
        print(f"  → {name}: extracting (this is the slow step)…")
        try:
            func = getattr(importlib.import_module(mod_name), func_name)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Could not import {mod_name}.{func_name} ({exc}). Run from the "
                f"project root, or extract once so {csv} exists.")
        func(longitudinal=True)


def load_clinical(path: str):
    """Load the longitudinal clinical CSV and return two caliber_code lookups
    (to patient_id and to visit) plus a per-(patient, visit) metadata table with
    symptom state, severity scores and language."""
    clin = pd.read_csv(path)
    clin["caliber_code"] = clin["caliber_code"].astype(str).str.lower().str.strip()
    clin["visit"] = clin["visit"].astype(str).str.strip()
    clin["patient_id"] = clin["id"].astype(str).str.lower().str.strip()

    code_to_patient = (clin[["caliber_code", "patient_id"]].dropna()
                       .set_index("caliber_code")["patient_id"].to_dict())
    code_to_visit = (clin[["caliber_code", "visit"]].dropna()
                     .set_index("caliber_code")["visit"].to_dict())

    cols = [c for c in ["patient_id", "visit", "symptom_state", "ymrs", "hdrs", "language"]
            if c in clin.columns]
    lookup = clin[cols].drop_duplicates(subset=["patient_id", "visit"]).copy()
    return code_to_patient, code_to_visit, lookup


def load_methodology(name, cfg, code_to_patient, code_to_visit):
    """Load one methodology's cached session CSV, map each caliber_code to its
    patient_id and visit, select the configured (or auto-detected numeric)
    feature columns, and collapse within-session multiplicity by averaging.
    Returns the tidied frame and its feature list, or None if the CSV is
    missing."""
    csv = Path(cfg["csv"])
    if not csv.exists():
        print(f"  ✗ {name}: {csv} not found — skipped.")
        return None

    df = pd.read_csv(csv)
    df["caliber_code"] = df["caliber_code"].astype(str).str.lower().str.strip()
    df["patient_id"] = df["caliber_code"].map(code_to_patient)
    df["visit"] = df["caliber_code"].map(code_to_visit)
    n_unmapped = df["patient_id"].isna().sum()
    if n_unmapped:
        print(f"  ⚠ {name}: {n_unmapped} sessions unmapped to patient_id — dropped")
    df = df.dropna(subset=["patient_id"]).copy()

    if cfg["features"]:
        feats = [f for f in cfg["features"] if f in df.columns]
    else:
        feats = [c for c in df.columns
                 if c not in _META and pd.api.types.is_numeric_dtype(df[c])
                 and df[c].nunique() > 1]

    keys = ["patient_id", "visit"] + (["task"] if "task" in df.columns else [])

    # Average over within-session multiplicity (e.g. speech-graph window configs).
    out = df[keys + feats]
    if out.duplicated(keys).any():
        out = out.groupby(keys, as_index=False)[feats].mean()

    print(f"  ✓ {name}: {len(feats)} features, {len(out)} rows")
    return out, feats


def build_long_features():
    """Build the merged longitudinal dataset: ensure each family's session CSV is
    extracted, merge them on patient_id/visit/task, join clinical metadata,
    optionally collapse across tasks per COLLAPSE_TASKS, and write the combined
    long_features.csv."""
    print("STEP 1 — extraction (cached families are skipped)")
    ensure_extracted(force=FORCE_EXTRACT)

    print("\nSTEP 2 — merging cached per-methodology CSVs")
    code_to_patient, code_to_visit, lookup = load_clinical(CLINICAL_CSV)

    merged = None
    seen_feats: set[str] = set()
    for name, cfg in METHODOLOGIES.items():
        res = load_methodology(name, cfg, code_to_patient, code_to_visit)
        if res is None:
            continue
        df, feats = res
        new_feats = [f for f in feats if f not in seen_feats]
        keys = [k for k in ["patient_id", "visit", "task"] if k in df.columns]
        df = df[keys + new_feats]
        seen_feats.update(new_feats)

        if merged is None:
            merged = df
        else:
            join_keys = [k for k in keys if k in merged.columns]
            merged = merged.merge(df, on=join_keys, how="outer")

    if merged is None:
        raise SystemExit("No methodology CSVs found — extraction may have failed.")

    merged = merged.merge(lookup, on=["patient_id", "visit"], how="left")
    merged = merged.rename(columns={"symptom_state": "mood_state"})

    if COLLAPSE_TASKS == "mean" and "task" in merged.columns:
        feat_cols = [c for c in merged.columns if c not in _META]
        clin_cols = [c for c in ["mood_state", "language", "ymrs", "hdrs"]
                     if c in merged.columns]
        merged = (merged.groupby(["patient_id", "visit"], as_index=False)
                  .agg({**{c: "mean" for c in feat_cols},
                        **{c: "first" for c in clin_cols}}))
    elif isinstance(COLLAPSE_TASKS, str) and "task" in merged.columns:
        merged = merged[merged["task"] == COLLAPSE_TASKS].copy()

    merged["session_id"] = merged["visit"]

    front = [c for c in ["patient_id", "session_id", "visit", "task", "mood_state",
                         "language", "word_count", "ymrs", "hdrs"] if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    merged.to_csv(OUT_CSV, index=False)
    n_feat = len([c for c in rest if c not in _META])
    print(f"\n  → {OUT_CSV}")
    print(f"     {merged['patient_id'].nunique()} patients | {len(merged)} rows | "
          f"{n_feat} feature columns")
    if "task" in merged.columns:
        print(f"     tasks kept: {sorted(merged['task'].dropna().unique())} "
              f"(rq3 script adds task as a covariate)")


if __name__ == "__main__":
    build_long_features()