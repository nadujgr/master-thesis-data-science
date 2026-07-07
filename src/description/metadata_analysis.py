"""
Dataset overview statistics for the CALIBER Audio Study. Computes Table-1 style
summaries for three views of the clinical metadata: all sessions, the
unique-patient analysis set, and the longitudinal subset (patients with 2+
visits). Analyses use a 3-level analysis_state (euthymia / depression / mania)
with score-derived "mixed" folded into "mania" per the Methods, while the raw
4-level symptom_state is kept for the agreement crosstab. The longitudinal view
adds transition matrices, per-transition symptom changes, inter-visit intervals,
within-patient variability, the depression-to-euthymia paired analysis, and
clinically significant changes. Reads the labelled, unique, and longitudinal
metadata CSVs (plus the speech-graph sessions file defining the F1 sample) and
writes results/analysis_results.json and several CSV tables.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    AUDIO_DATA_LABELLED,
    AUDIO_DATA_UNIQUE,
    AUDIO_DATA_LONGITUDINAL,
    DATA_OVERVIEW_RESULTS_DIR,
    RESULTS_DIR,
)

# Longitudinal F1 sample is defined by the speech-graph sessions file (usable
# emotion-task transcripts), mirroring the RQ3 pipeline.
SG_SESSIONS_CSV       = Path(RESULTS_DIR) / "longitudinal" / "speech_graphs" / "speech_graphs_sessions.csv"
SG_TASK               = "emotion"
SG_PRIMARY_WINDOW     = "results_split_10_1_100rand"
SG_PRIMARY_EXPERIMENT = "emotion_w10s1"
SG_MIN_SESSIONS       = 2             # F1 threshold

# Filtered longitudinal file written for downstream plots to consume.
AUDIO_DATA_LONGITUDINAL_F1 = Path(AUDIO_DATA_LONGITUDINAL).with_name(
    Path(AUDIO_DATA_LONGITUDINAL).stem + "_f1.csv")

# ── Constants ────────────────────────────────────────────────────────────────
# 3-level grouping used for Table 1 and downstream plots/tests.
# Score-derived "mixed" is collapsed into "mania" here (see Methods).
ANALYSIS_GROUPS = ["euthymia", "depression", "mania"]

# 4-level raw symptom_state — kept for the agreement matrix and any
# later sub-analysis needing a pure-mania vs mixed split.
SYMPTOM_STATES = ["euthymia", "depression", "mania", "mixed"]
EPISODE_ORDER   = ["depression", "euthymia", "mania"]
CLINICAL_SCALES = ["ymrs", "hdrs", "panss_p", "panss_n", "panss_g"]
RESULTS_FILE    = "analysis_results.json"

# Minimum YMRS/HDRS point change regarded as clinically significant.
CLINICAL_CHANGE_THRESHOLD = 5


# ── Public entry points ───────────────────────────────────────────────────────

def compute_full_dataset_overview():
    """Overview of all sessions (multiple visits per patient possible)."""
    print("  Loading full labelled dataset (all sessions)...")
    df = pd.read_csv(AUDIO_DATA_LABELLED)
    print(f"    {len(df)} sessions, {df['id'].nunique()} unique patients.")
    
    overview = _build_overview(df, level="session")
    overview["n_patients"]      = int(df["id"].nunique())
    overview["visit_structure"] = _visit_structure(df)
    
    _save_overview("full_dataset_overview", overview)


def compute_unique_dataset_overview():
    """Overview of the final analysis dataset (one session per patient)."""
    print("  Loading unique-patient dataset (one session per patient)...")
    df = pd.read_csv(AUDIO_DATA_UNIQUE)
    print(f"    {len(df)} patients.")
    
    overview = _build_overview(df, level="patient")
    
    _save_overview("unique_dataset_overview", overview)

def _f1_usable_codes(verbose: bool = True) -> tuple[set, set]:
    """Return (usable caliber_codes, F1 patient-id prefixes) from the speech-graph
    sessions file for the emotion task at the primary window, keeping only patients
    with >= SG_MIN_SESSIONS usable sessions."""
    if not SG_SESSIONS_CSV.exists():
        raise FileNotFoundError(
            f"Speech-graph sessions file not found: {SG_SESSIONS_CSV}. "
            "Run the speech-graph extraction before the longitudinal overview.")

    sg = pd.read_csv(SG_SESSIONS_CSV)
    rename = {}
    if "caliber_code"  in sg.columns: rename["caliber_code"]  = "episode_id"
    if "symptom_state" in sg.columns: rename["symptom_state"] = "mood_state"
    if "task_set"      in sg.columns: rename["task_set"]      = "task"
    sg = sg.rename(columns=rename)

    sg["episode_id"] = sg["episode_id"].astype(str).str.strip().str.lower()
    sg["mood_state"] = (sg["mood_state"].astype(str).str.strip().str.lower()
                          .replace("mixed", "mania"))
    sg = sg[sg["mood_state"].isin(ANALYSIS_GROUPS)]
    sg["patient_id"] = sg["episode_id"].str.extract(r'^(\d+)_')[0]

    if "task" in sg.columns:
        sg = sg[sg["task"] == SG_TASK]
    for col, val in [("window", SG_PRIMARY_WINDOW),
                     ("experiment", SG_PRIMARY_EXPERIMENT)]:
        if col in sg.columns and (sg[col] == val).any():
            sg = sg[sg[col] == val]
            break

    sg = sg.drop_duplicates("episode_id")                       # one row / session
    counts      = sg.groupby("patient_id")["episode_id"].nunique()
    f1_patients = set(counts[counts >= SG_MIN_SESSIONS].index)
    usable      = set(sg.loc[sg["patient_id"].isin(f1_patients), "episode_id"])

    if verbose:
        print(f"    F1 (speech-graph file): {len(f1_patients)} patients, "
              f"{len(usable)} usable emotion sessions")
        print(f"      patients: {sorted(f1_patients, key=int)}")
    return usable, f1_patients


def apply_longitudinal_filter(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Restrict a clinical longitudinal frame to the F1 sample (patients with >=2
    usable emotion-task speech-graph sessions), matching on caliber_code.
    """
    usable, f1_patients = _f1_usable_codes(verbose)
    if "caliber_code" not in df.columns:
        print("    ⚠ no caliber_code column — cannot apply F1 filter; "
              "returning data unfiltered.")
        return df

    code = df["caliber_code"].astype(str).str.strip().str.lower()
    out  = df[code.isin(usable)].copy()
    if verbose:
        print(f"    Applied F1: {len(df)} → {len(out)} sessions, "
              f"{out['id'].nunique()} patients")
    return out


def compute_longitudinal_dataset_overview():
    """Overview of the longitudinal subset (patients with 2+ visits): standard
    Table 1 stats plus transition matrix, per-transition symptom changes,
    inter-visit intervals, within-patient variability, the depression→euthymia
    paired analysis, and clinically significant changes. Also writes CSV files."""
    print("\n" + "="*80)
    print("  LONGITUDINAL DATASET OVERVIEW")
    print("="*80)
    print("  Loading longitudinal dataset (patients with 2+ visits)...")

    df = pd.read_csv(AUDIO_DATA_LONGITUDINAL)
    print(f"    {len(df)} sessions from {df['id'].nunique()} patients (raw, 2+ visits).")

    # Restrict to the F1 sample (usable emotion transcripts).
    df = apply_longitudinal_filter(df, verbose=True)

    # Persist the filtered sample so downstream plots use the same population.
    AUDIO_DATA_LONGITUDINAL_F1.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIO_DATA_LONGITUDINAL_F1, index=False)
    print(f"  ✓ filtered longitudinal → {AUDIO_DATA_LONGITUDINAL_F1}")

    overview = _build_overview(df, level="session")
    overview["n_patients"] = int(df["id"].nunique())
    overview["visit_structure"] = _visit_structure(df)

    print("\n  Computing longitudinal-specific analyses...")

    print("    • Transition matrix...")
    overview["transition_matrix"], trans_df = _transition_matrix(df)

    print("    • Symptom changes by transition...")
    overview["symptom_changes"], symptom_changes_df = _symptom_changes_by_transition(df)

    print("    • Time between visits...")
    overview["time_between_visits"] = _time_between_visits(df)

    print("    • Within-patient variability...")
    overview["within_patient_variability"] = _within_patient_variability(df)

    print("    • Depression→euthymia transitions...")
    overview["depression_to_euthymia"] = _depression_to_euthymia_details(df)

    print("    • Clinically significant changes...")
    overview["clinically_significant_changes"] = _clinically_significant_changes(df)

    print("    • Per-patient summary...")
    patient_summary_df = _patient_summary(df)

    _save_overview("longitudinal_dataset_overview", overview)

    csv_dir = Path(DATA_OVERVIEW_RESULTS_DIR) / "longitudinal_csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    trans_df.to_csv(csv_dir / "longitudinal_transition_matrix.csv", index=False)
    print(f"  ✓ CSV → {csv_dir / 'longitudinal_transition_matrix.csv'}")

    symptom_changes_df.to_csv(csv_dir / "longitudinal_symptom_changes.csv", index=False)
    print(f"  ✓ CSV → {csv_dir / 'longitudinal_symptom_changes.csv'}")

    patient_summary_df.to_csv(csv_dir / "longitudinal_patient_summary.csv", index=False)
    print(f"  ✓ CSV → {csv_dir / 'longitudinal_patient_summary.csv'}")

    _print_longitudinal_summary(overview)


# ── Core computation (shared) ─────────────────────────────────────────────────

def _build_overview(df: pd.DataFrame, level: str) -> dict:
    """Assemble the shared overview dictionary (totals, Table 1, raw-by-state
    lists, agreement crosstab, and episode counts) for a metadata frame at the
    given level ("session" or "patient")."""
    df = df.copy()
    # Collapsed analysis grouping: mixed → mania.
    df["analysis_state"] = df["symptom_state"].replace({"mixed": "mania"})
    
    return {
        "n_total":      int(len(df)),
        "level":        level,
        "table1":       _table1(df),
        "raw_by_state": _raw_by_state(df),
        "agreement":    _agreement_matrix(df),
        "episode_counts": df["episode_type"].value_counts().to_dict(),
    }


def _table1(df: pd.DataFrame) -> dict:
    """Publication-style Table 1 per analysis group (euthymia / depression /
    mania, where mania = pure mania + mixed) plus an "all" column, with
    demographics, clinical-scale descriptives, episode counts, and the raw
    4-level symptom_state breakdown."""
    table = {}
    groups = [(g, df[df["analysis_state"] == g]) for g in ANALYSIS_GROUPS]
    groups.append(("all", df))
    
    for group, sub in groups:
        if sub.empty:
            continue
        
        n = len(sub)
        col = {"n": n}

        if "age" in sub.columns and sub["age"].notna().any():
            col["age"] = {
                "mean": round(float(sub["age"].mean()), 1),
                "sd":   round(float(sub["age"].std()),  1),
            }
        
        col["sex"] = _categorical_counts(sub, "sex", n)

        if "language" in sub.columns:
            col["language"] = _categorical_counts(sub, "language", n)

        for scale in CLINICAL_SCALES:
            if scale not in sub.columns:
                continue
            s = sub[scale].dropna()
            if s.empty:
                continue
            col[scale] = {
                "n":      int(len(s)),
                "mean":   round(float(s.mean()),         2),
                "sd":     round(float(s.std()),          2),
                "median": round(float(s.median()),       1),
                "q1":     round(float(s.quantile(0.25)), 1),
                "q3":     round(float(s.quantile(0.75)), 1),
                "min":    round(float(s.min()),          1),
                "max":    round(float(s.max()),          1),
            }
        
        col["episode_type"] = _categorical_counts(sub, "episode_type", n)
        if "episode_time" in sub.columns:
            col["episode_time"] = _categorical_counts(sub, "episode_time", n)

        # Surfaces the pure-mania vs mixed split inside the collapsed "mania" group.
        col["symptom_state"] = _categorical_counts(sub, "symptom_state", n)
        
        table[group] = col
    
    return table


def _categorical_counts(df: pd.DataFrame, col: str, n: int) -> dict:
    """{value: {n, pct}} for one column, ignoring NaN."""
    if col not in df.columns or n == 0:
        return {}
    return {
        str(k): {"n": int(v), "pct": round(v / n * 100, 1)}
        for k, v in df[col].dropna().value_counts().items()
    }


def _raw_by_state(df: pd.DataFrame) -> dict:
    """Raw value lists per analysis group — for boxplots / violins downstream."""
    out = {}
    for group in ANALYSIS_GROUPS:
        sub = df[df["analysis_state"] == group]
        if sub.empty:
            continue
        block = {}
        if "age" in sub.columns:
            block["age"] = sub["age"].dropna().tolist()
        for scale in CLINICAL_SCALES:
            if scale in sub.columns:
                block[scale] = sub[scale].dropna().tolist()
        out[group] = block
    return out


def _agreement_matrix(df: pd.DataFrame) -> dict:
    """Crosstab of clinical episode_type (rows) × raw 4-level symptom_state
    (cols). Uses the original symptom_state so the pure-mania vs mixed distinction
    stays visible — this table justifies the merge in the Methods."""
    return (
        pd.crosstab(df["episode_type"], df["symptom_state"])
        .reindex(index=EPISODE_ORDER, columns=SYMPTOM_STATES, fill_value=0)
        .to_dict(orient="index")
    )


def _visit_structure(df: pd.DataFrame) -> dict:
    """{n_visits: n_patients} — only meaningful for the session-level dataset."""
    counts = df.groupby("id").size().value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


# ── Longitudinal-specific functions ───────────────────────────────────────────

def _transition_matrix(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """State transition matrix for consecutive visits. Returns (stats dict,
    DataFrame for CSV export)."""
    df = df.copy()
    df["symptom_state"] = df["symptom_state"].replace("mixed", "mania")
    df = df[df["symptom_state"].isin(ANALYSIS_GROUPS)]
    df = df.sort_values(["id", "visit"])
    
    transitions = []
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        for i in range(len(patient_visits) - 1):
            v1 = patient_visits.iloc[i]
            v2 = patient_visits.iloc[i + 1]
            
            transitions.append({
                'patient_id': patient_id,
                'from_state': v1["symptom_state"],
                'to_state': v2["symptom_state"],
                'from_visit': v1["visit"],
                'to_visit': v2["visit"],
                'changed': int(v1["symptom_state"] != v2["symptom_state"])
            })
    
    trans_df = pd.DataFrame(transitions)

    matrix = {}
    for s1 in SYMPTOM_STATES:
        matrix[s1] = {s2: 0 for s2 in SYMPTOM_STATES}
    
    for _, row in trans_df.iterrows():
        s1, s2 = row['from_state'], row['to_state']
        if s1 in SYMPTOM_STATES and s2 in SYMPTOM_STATES:
            matrix[s1][s2] += 1
    
    n_transitions = len(trans_df)
    n_changes = int(trans_df['changed'].sum())
    
    stats = {
        "matrix": matrix,
        "n_transitions": n_transitions,
        "n_state_changes": n_changes,
        "pct_state_changes": round(n_changes / n_transitions * 100, 1) if n_transitions > 0 else 0
    }
    
    return stats, trans_df


def _symptom_changes_by_transition(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """YMRS and HDRS changes for each transition type. Returns (stats dict,
    DataFrame for CSV export)."""
    df = df.sort_values(["id", "visit"])
    
    changes = []
    
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        for i in range(len(patient_visits) - 1):
            v1 = patient_visits.iloc[i]
            v2 = patient_visits.iloc[i + 1]
            
            transition_type = f"{v1['symptom_state']}_to_{v2['symptom_state']}"
            
            changes.append({
                'patient_id': patient_id,
                'transition_type': transition_type,
                'from_state': v1['symptom_state'],
                'to_state': v2['symptom_state'],
                'delta_ymrs': float(v2["ymrs"] - v1["ymrs"]) if pd.notna(v1["ymrs"]) and pd.notna(v2["ymrs"]) else np.nan,
                'delta_hdrs': float(v2["hdrs"] - v1["hdrs"]) if pd.notna(v1["hdrs"]) and pd.notna(v2["hdrs"]) else np.nan,
                'ymrs_v1': float(v1["ymrs"]) if pd.notna(v1["ymrs"]) else np.nan,
                'ymrs_v2': float(v2["ymrs"]) if pd.notna(v2["ymrs"]) else np.nan,
                'hdrs_v1': float(v1["hdrs"]) if pd.notna(v1["hdrs"]) else np.nan,
                'hdrs_v2': float(v2["hdrs"]) if pd.notna(v2["hdrs"]) else np.nan,
            })
    
    changes_df = pd.DataFrame(changes)

    summary = {}
    for trans_type in changes_df['transition_type'].unique():
        sub = changes_df[changes_df['transition_type'] == trans_type]
        
        ymrs_vals = sub['delta_ymrs'].dropna()
        hdrs_vals = sub['delta_hdrs'].dropna()
        
        summary[trans_type] = {
            "n": len(sub),
            "ymrs": {
                "mean": round(float(ymrs_vals.mean()), 2) if len(ymrs_vals) > 0 else None,
                "median": round(float(ymrs_vals.median()), 2) if len(ymrs_vals) > 0 else None,
                "sd": round(float(ymrs_vals.std()), 2) if len(ymrs_vals) > 0 else None,
            },
            "hdrs": {
                "mean": round(float(hdrs_vals.mean()), 2) if len(hdrs_vals) > 0 else None,
                "median": round(float(hdrs_vals.median()), 2) if len(hdrs_vals) > 0 else None,
                "sd": round(float(hdrs_vals.std()), 2) if len(hdrs_vals) > 0 else None,
            }
        }
    
    return summary, changes_df


def _time_between_visits(df: pd.DataFrame) -> dict:
    """Distribution of time between consecutive visits (in months, from age)."""
    df = df.sort_values(["id", "visit"])
    
    intervals = []
    
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        if "age" not in patient_visits.columns:
            continue
        
        for i in range(len(patient_visits) - 1):
            age1 = patient_visits.iloc[i]["age"]
            age2 = patient_visits.iloc[i + 1]["age"]
            
            if pd.notna(age1) and pd.notna(age2):
                months = (age2 - age1) * 12
                intervals.append(float(months))
    
    if not intervals:
        return {"n": 0}
    
    return {
        "n": len(intervals),
        "mean_months": round(float(np.mean(intervals)), 1),
        "median_months": round(float(np.median(intervals)), 1),
        "sd_months": round(float(np.std(intervals)), 1),
        "min_months": round(float(np.min(intervals)), 1),
        "max_months": round(float(np.max(intervals)), 1),
        "quartiles": {
            "q25": round(float(np.percentile(intervals, 25)), 1),
            "q75": round(float(np.percentile(intervals, 75)), 1)
        }
    }


def _within_patient_variability(df: pd.DataFrame) -> dict:
    """Within-patient SD for clinical scales (how much individuals vary)."""
    variability = {}
    
    for scale in CLINICAL_SCALES:
        if scale not in df.columns:
            continue
        
        patient_sds = []
        
        for patient_id in df["id"].unique():
            patient_data = df[df["id"] == patient_id][scale].dropna()
            
            if len(patient_data) >= 2:
                patient_sds.append(float(patient_data.std()))
        
        if patient_sds:
            variability[scale] = {
                "mean_within_patient_sd": round(float(np.mean(patient_sds)), 2),
                "median_within_patient_sd": round(float(np.median(patient_sds)), 2),
                "n_patients": len(patient_sds)
            }
    
    return variability


def _depression_to_euthymia_details(df: pd.DataFrame) -> dict:
    """Details about depression→euthymia transitions (primary paired analysis)."""
    df = df.sort_values(["id", "visit"])
    
    dep_to_euth_pairs = []
    
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        # First depression, then first euthymia after it.
        dep_visits = patient_visits[patient_visits["symptom_state"] == "depression"]
        euth_visits = patient_visits[patient_visits["symptom_state"] == "euthymia"]
        
        if len(dep_visits) > 0 and len(euth_visits) > 0:
            first_dep_idx = dep_visits.index[0]
            euth_after = euth_visits[euth_visits.index > first_dep_idx]
            
            if len(euth_after) > 0:
                dep_visit = dep_visits.iloc[0]
                euth_visit = euth_after.iloc[0]
                
                dep_to_euth_pairs.append({
                    "patient_id": patient_id,
                    "ymrs_change": float(euth_visit["ymrs"] - dep_visit["ymrs"]) if pd.notna(dep_visit["ymrs"]) and pd.notna(euth_visit["ymrs"]) else None,
                    "hdrs_change": float(euth_visit["hdrs"] - dep_visit["hdrs"]) if pd.notna(dep_visit["hdrs"]) and pd.notna(euth_visit["hdrs"]) else None,
                })
    
    if not dep_to_euth_pairs:
        return {"n": 0}
    
    ymrs_changes = [p["ymrs_change"] for p in dep_to_euth_pairs if p["ymrs_change"] is not None]
    hdrs_changes = [p["hdrs_change"] for p in dep_to_euth_pairs if p["hdrs_change"] is not None]
    
    return {
        "n": len(dep_to_euth_pairs),
        "patients": [p["patient_id"] for p in dep_to_euth_pairs],
        "ymrs_change": {
            "mean": round(float(np.mean(ymrs_changes)), 2) if ymrs_changes else None,
            "median": round(float(np.median(ymrs_changes)), 2) if ymrs_changes else None,
            "sd": round(float(np.std(ymrs_changes)), 2) if ymrs_changes else None,
        },
        "hdrs_change": {
            "mean": round(float(np.mean(hdrs_changes)), 2) if hdrs_changes else None,
            "median": round(float(np.median(hdrs_changes)), 2) if hdrs_changes else None,
            "sd": round(float(np.std(hdrs_changes)), 2) if hdrs_changes else None,
        }
    }


def _clinically_significant_changes(df: pd.DataFrame) -> dict:
    """Count patients with clinically significant symptom changes (Δ ≥ 5 points)."""
    df = df.sort_values(["id", "visit"])
    
    patients_with_ymrs_change = []
    patients_with_hdrs_change = []
    
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        if "ymrs" in patient_visits.columns:
            ymrs_vals = patient_visits["ymrs"].dropna()
            if len(ymrs_vals) >= 2:
                ymrs_range = ymrs_vals.max() - ymrs_vals.min()
                if ymrs_range >= CLINICAL_CHANGE_THRESHOLD:
                    patients_with_ymrs_change.append(patient_id)

        if "hdrs" in patient_visits.columns:
            hdrs_vals = patient_visits["hdrs"].dropna()
            if len(hdrs_vals) >= 2:
                hdrs_range = hdrs_vals.max() - hdrs_vals.min()
                if hdrs_range >= CLINICAL_CHANGE_THRESHOLD:
                    patients_with_hdrs_change.append(patient_id)
    
    return {
        "ymrs_change_ge5": {
            "n": len(patients_with_ymrs_change),
            "patients": patients_with_ymrs_change
        },
        "hdrs_change_ge5": {
            "n": len(patients_with_hdrs_change),
            "patients": patients_with_hdrs_change
        },
        "either_scale": {
            "n": len(set(patients_with_ymrs_change + patients_with_hdrs_change))
        }
    }


def _patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient summary for CSV export."""
    df = df.sort_values(["id", "visit"])
    
    summaries = []
    
    for patient_id in df["id"].unique():
        patient_visits = df[df["id"] == patient_id].sort_values("visit")
        
        states = patient_visits["symptom_state"].tolist()
        state_sequence = " → ".join(states)
        unique_states = patient_visits["symptom_state"].nunique()

        ymrs_vals = patient_visits["ymrs"].dropna()
        hdrs_vals = patient_visits["hdrs"].dropna()
        
        summaries.append({
            'patient_id': patient_id,
            'n_visits': len(patient_visits),
            'state_sequence': state_sequence,
            'n_unique_states': unique_states,
            'state_changed': int(unique_states > 1),
            'ymrs_min': float(ymrs_vals.min()) if len(ymrs_vals) > 0 else np.nan,
            'ymrs_max': float(ymrs_vals.max()) if len(ymrs_vals) > 0 else np.nan,
            'ymrs_range': float(ymrs_vals.max() - ymrs_vals.min()) if len(ymrs_vals) > 0 else np.nan,
            'hdrs_min': float(hdrs_vals.min()) if len(hdrs_vals) > 0 else np.nan,
            'hdrs_max': float(hdrs_vals.max()) if len(hdrs_vals) > 0 else np.nan,
            'hdrs_range': float(hdrs_vals.max() - hdrs_vals.min()) if len(hdrs_vals) > 0 else np.nan,
            'sex': patient_visits.iloc[0]['sex'] if 'sex' in patient_visits.columns else np.nan,
            'language': patient_visits.iloc[0]['language'] if 'language' in patient_visits.columns else np.nan,
        })
    
    return pd.DataFrame(summaries)


def _print_longitudinal_summary(overview: dict) -> None:
    """Pretty print longitudinal overview to console."""
    print(f"\n{'='*80}")
    print("  LONGITUDINAL OVERVIEW SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n  Recordings: {overview['n_total']}")
    print(f"  Patients: {overview['n_patients']}")

    print(f"\n  Visit Structure:")
    for n_visits, n_patients in sorted(overview['visit_structure'].items(),
                                      key=lambda x: int(x[0]), reverse=True):
        print(f"    {n_patients} patients × {n_visits} visits")

    trans = overview['transition_matrix']
    print(f"\n  State Transitions:")
    print(f"    {trans['n_transitions']} consecutive visit pairs")
    print(f"    {trans['n_state_changes']} state changes ({trans['pct_state_changes']}%)")

    dep_euth = overview['depression_to_euthymia']
    print(f"\n  Depression → Euthymia (PRIMARY ANALYSIS):")
    print(f"    n = {dep_euth['n']} patients")
    if dep_euth['n'] > 0:
        print(f"    HDRS: Δ = {dep_euth['hdrs_change']['mean']}±{dep_euth['hdrs_change']['sd']}")
        print(f"    YMRS: Δ = {dep_euth['ymrs_change']['mean']}±{dep_euth['ymrs_change']['sd']}")

    clin = overview['clinically_significant_changes']
    print(f"\n  Clinically Significant Changes (Δ ≥ 5):")
    print(f"    YMRS: {clin['ymrs_change_ge5']['n']} patients")
    print(f"    HDRS: {clin['hdrs_change_ge5']['n']} patients")
    print(f"    Either: {clin['either_scale']['n']} patients")
    
    print(f"\n{'='*80}\n")

def _build_table1_rows(t: dict, groups: list, wv: dict = None,
                        dep_euth: dict = None) -> list:
    """Flatten a computed table1 dict into publication-ready rows (one per
    variable, with a column per group plus a total). When within-patient
    variability and depression-to-euthymia details are supplied the rows include
    the longitudinal extras; otherwise the pure-mania vs mixed breakdown is added."""
    rows = []

    def _row(variable, *cells):
        rows.append({"variable": variable,
                     **{g: cells[i] for i, g in enumerate(groups)},
                     "total": cells[len(groups)]})

    def _scale_iqr(grp, scale):
        s = t[grp].get(scale)
        return f"{s['median']} ({s['q1']}–{s['q3']})" if s else ""

    def _cat(grp, col, cat):
        s = t[grp].get(col, {}).get(cat)
        return f"{s['n']} ({s['pct']}%)" if s else "0"

    _row("N", *[t[g]["n"] for g in groups], t["all"]["n"])
    _row("Age, mean (SD)",
         *[f"{t[g]['age']['mean']} ({t[g]['age']['sd']})" for g in groups],
         f"{t['all']['age']['mean']} ({t['all']['age']['sd']})")
    _row("Female, N (%)", *[_cat(g, "sex", "female") for g in groups], _cat("all", "sex", "female"))
    _row("Male, N (%)",   *[_cat(g, "sex", "male")   for g in groups], _cat("all", "sex", "male"))
    _row("Catalan, N (%)", *[_cat(g, "language", "catalan") for g in groups], _cat("all", "language", "catalan"))
    _row("Spanish, N (%)", *[_cat(g, "language", "spanish") for g in groups], _cat("all", "language", "spanish"))
    _row("English, N (%)", *[_cat(g, "language", "english") for g in groups], _cat("all", "language", "english"))
    _row("YMRS, median (IQR)", *[_scale_iqr(g, "ymrs") for g in groups], _scale_iqr("all", "ymrs"))
    _row("HDRS-17, median (IQR)", *[_scale_iqr(g, "hdrs") for g in groups], _scale_iqr("all", "hdrs"))

    if wv is None:
        # Unique dataset: add mania subgroup breakdown.
        _row("Pure mania, N (%)",
             *[_cat(g, "symptom_state", "mania") if g == "mania" else "—" for g in groups],
             _cat("all", "symptom_state", "mania"))
        _row("Mixed state, N (%)",
             *[_cat(g, "symptom_state", "mixed") if g == "mania" else "—" for g in groups],
             _cat("all", "symptom_state", "mixed"))
    else:
        # Longitudinal dataset: add within-patient variability and paired analysis.
        for scale, label in [("ymrs", "YMRS within-patient SD, mean"),
                              ("hdrs", "HDRS-17 within-patient SD, mean")]:
            v = wv.get(scale, {}).get("mean_within_patient_sd", "")
            _row(label, *["—"] * len(groups), str(v) if v else "")
        n_pairs = dep_euth.get("n", 0)
        hdrs_change = dep_euth.get("hdrs_change", {})
        change_str = (f"{hdrs_change['mean']} (SD={hdrs_change['sd']})"
                      if hdrs_change.get("mean") is not None else "")
        _row("Depression→Euthymia pairs, N", *["—"] * len(groups), str(n_pairs))
        _row("HDRS-17 change (dep→euth), mean (SD)", *["—"] * len(groups), change_str)

    return rows

def generate_table1_csv() -> None:
    """Read back analysis_results.json and write the unique and longitudinal
    Table 1 CSVs from the stored overviews."""
    results_path = Path(DATA_OVERVIEW_RESULTS_DIR) / RESULTS_FILE
    if not results_path.exists():
        print("  ✗ analysis_results.json not found — run compute_*_overview() first.")
        return

    with results_path.open() as f:
        data = json.load(f)

    if "unique_dataset_overview" in data:
        t = data["unique_dataset_overview"]["table1"]
        groups = [g for g in ANALYSIS_GROUPS if g in t]
        rows = _build_table1_rows(t, groups)
        pd.DataFrame(rows).to_csv(
            Path(DATA_OVERVIEW_RESULTS_DIR) / "table1_unique.csv", index=False)
        print(f"  ✓ CSV → {Path(DATA_OVERVIEW_RESULTS_DIR) / 'table1_unique.csv'}")

    if "longitudinal_dataset_overview" in data:
        t = data["longitudinal_dataset_overview"]["table1"]
        wv = data["longitudinal_dataset_overview"].get("within_patient_variability", {})
        dep_euth = data["longitudinal_dataset_overview"].get("depression_to_euthymia", {})
        groups = [g for g in ANALYSIS_GROUPS if g in t]
        rows = _build_table1_rows(t, groups, wv=wv, dep_euth=dep_euth)
        pd.DataFrame(rows).to_csv(
            Path(DATA_OVERVIEW_RESULTS_DIR) / "table1_longitudinal.csv", index=False)
        print(f"  ✓ CSV → {Path(DATA_OVERVIEW_RESULTS_DIR) / 'table1_longitudinal.csv'}")


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save_overview(key: str, overview: dict) -> None:
    results_path = Path(DATA_OVERVIEW_RESULTS_DIR) / RESULTS_FILE
    Path(DATA_OVERVIEW_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    existing = {}
    if results_path.exists():
        with results_path.open() as f:
            existing = json.load(f)
    
    existing[key] = overview
    
    with results_path.open("w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"  ✓ JSON → {results_path} (key: {key})")

def run_metadata_analysis() -> None:
    """Entry point: compute the full, unique, and longitudinal overviews and then
    write the Table 1 CSVs."""
    compute_full_dataset_overview()
    compute_unique_dataset_overview()
    compute_longitudinal_dataset_overview()
    generate_table1_csv()