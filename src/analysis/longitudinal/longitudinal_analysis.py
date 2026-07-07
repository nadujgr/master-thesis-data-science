"""
RQ3 — within-person mood-state effects on speech graph structure (emotion task).
Reads results/longitudinal/speech_graphs/speech_graphs_sessions.csv, extracts the
true patient_id from each caliber_code prefix, and fits patient-grouped mixed
models. Outputs per-feature tables for mood effects, ICC, severity effects, a
mania-subset check, and per-visit trajectories.

Each effect is fitted at nested tiers (unadjusted / +language / +word_count) so
covariate influence is visible directly. Severity scores (YMRS/HDRS) are
person-mean centered (_wp); graph features are population z-scored (_z).
Benjamini–Hochberg FDR is applied separately within each model tier.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

from src.config import RESULTS_DIR
_RES = Path(RESULTS_DIR)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

INPUT_CSV  = str(_RES / "longitudinal" / "speech_graphs" / "speech_graphs_sessions.csv")
OUTPUT_DIR = _RES / "longitudinal" / "speech_graphs_emotion"

TASK               = "emotion"
PRIMARY_EXPERIMENT = "emotion_w10s1"
PRIMARY_WINDOW     = "results_split_10_1_100rand"
MOOD_REFERENCE     = "euthymia"
MOOD_LEVELS        = ["euthymia", "depression", "mania"]

SGA_METRICS = [
    "LSC",  "Density",
    "ASP", "CC", "RE", "L1", "L2",
]

# Between-patient metrics, used to order the trajectory CSV
T1_METRICS = [
    "LSC",  "Density",
    "ASP", "CC", "RE", "L1", "L2",
]
MIN_SESSIONS_PER_PATIENT = 2
MIN_OBS_FOR_MODEL        = 6
FDR_ALPHA                = 0.05
BETA_FLOOR               = 0.20

MOOD_TERM   = "C(mood_state, Treatment(reference='euthymia'))"
_OPTIMIZERS = ["lbfgs", "bfgs", "nm", "powell"]

MOOD_TIERS = ("unadj", "lang_adj", "full_adj")
SEV_TIERS  = ("unadj", "lang_adj")


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING + FILTERING
# ═══════════════════════════════════════════════════════════════════

def _extract_patient_id(caliber_code: pd.Series) -> pd.Series:
    """'107_mde_bd_acute' → '107'  (numeric prefix before first underscore)."""
    return caliber_code.astype(str).str.extract(r'^(\d+)_')[0]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns
    -------
    df_all    : F1 applied (≥2 sessions). Used for ICC.
    df_models : F1 + F2 applied. Used for mood_effects and severity_effects.
    df_mania  : F1 + F2 + has mania + has non-mania. Mania-subset / trajectories.
    available : list of SGA metrics with non-zero variance in this window.
    """
    raw = pd.read_csv(INPUT_CSV)

    # ── Column normalisation ──────────────────────────────────────
    rename = {}
    if "caliber_code"  in raw.columns: rename["caliber_code"]  = "episode_id"
    if "symptom_state" in raw.columns: rename["symptom_state"] = "mood_state"
    if "task_set"      in raw.columns: rename["task_set"]       = "task"
    raw = raw.rename(columns=rename)

    raw["episode_id"] = raw["episode_id"].astype(str).str.strip().str.lower()
    raw["mood_state"] = raw["mood_state"].astype(str).str.strip().str.lower()
    if "language" in raw.columns:
        raw["language"] = raw["language"].astype(str).str.strip().str.lower()

    # Extract real patient ID from the episode code
    raw["patient_id"] = _extract_patient_id(raw["episode_id"])

    # Fold mixed → mania, keep only valid mood states
    raw["mood_state"] = raw["mood_state"].replace("mixed", "mania")
    raw = raw[raw["mood_state"].isin(MOOD_LEVELS)].copy()

    # ── Task filter ───────────────────────────────────────────────
    if "task" in raw.columns:
        before = len(raw)
        raw = raw[raw["task"] == TASK].copy()
        print(f"  Restricted to task='{TASK}': {before} → {len(raw)} rows")

    # ── Primary window filter ─────────────────────────────────────
    filtered = False
    for col, val in [("window", PRIMARY_WINDOW), ("experiment", PRIMARY_EXPERIMENT)]:
        if col in raw.columns and (raw[col] == val).any():
            before = len(raw)
            raw = raw[raw[col] == val].copy()
            print(f"  Restricted to {col}='{val}': {before} → {len(raw)} rows")
            filtered = True
            break
    if not filtered:
        print("  ⚠ No window/experiment column matched — using all rows")

    # ── Z-score SGA metrics (population mean/SD for cross-metric comparability)
    for col in SGA_METRICS + ["word_count"]:
        if col in raw.columns:
            sd = raw[col].std(ddof=0)
            raw[col + "_z"] = (raw[col] - raw[col].mean()) / sd if sd else 0.0

    # ── Within-person centering for clinical severity scores ──────
    for score in ("ymrs", "hdrs"):
        if score in raw.columns:
            person_mean      = raw.groupby("patient_id")[score].transform("mean")
            raw[score + "_wp"] = raw[score] - person_mean
            print(f"  Within-person centering: {score}_wp created "
                  f"(range {raw[score + '_wp'].min():.1f} to "
                  f"{raw[score + '_wp'].max():.1f})")

    # ── Zero-variance filter ──────────────────────────────────────
    zero_var      = []
    available_all = []
    for f in SGA_METRICS:
        if f not in raw.columns:
            continue
        if raw[f].std() < 1e-10:
            zero_var.append(f)
        else:
            available_all.append(f)
    if zero_var:
        print(f"  Zero-variance filter: excluded {zero_var}")

    # ── F1: ≥ 2 sessions per patient ─────────────────────────────
    counts     = raw.groupby("patient_id").size()
    keep_f1    = counts[counts >= MIN_SESSIONS_PER_PATIENT].index
    dropped_f1 = counts[counts < MIN_SESSIONS_PER_PATIENT].index.tolist()
    if dropped_f1:
        print(f"  F1 (single-session): dropped patients {dropped_f1}")
    df_all = raw[raw["patient_id"].isin(keep_f1)].copy()

    df_all["mood_state"] = pd.Categorical(
        df_all["mood_state"],
        categories=[MOOD_REFERENCE] + [l for l in MOOD_LEVELS if l != MOOD_REFERENCE])

    print(f"  After F1: {df_all['patient_id'].nunique()} patients | "
          f"{len(df_all)} sessions")
    print("  States: " + " | ".join(
        f"{s}={int((df_all['mood_state'] == s).sum())}" for s in MOOD_LEVELS))

    # ── F2: ≥ 2 distinct mood states per patient ──────────────────
    state_counts = df_all.groupby("patient_id")["mood_state"].nunique()
    keep_f2      = state_counts[state_counts >= 2].index
    mono_pats    = state_counts[state_counts < 2].index.tolist()
    if mono_pats:
        mono_info = {
            p: df_all[df_all["patient_id"] == p]["mood_state"].unique().tolist()
            for p in mono_pats
        }
        print(f"  F2 (single-state): excluded from models (kept for ICC): {mono_info}")
    df_models = df_all[df_all["patient_id"].isin(keep_f2)].copy()
    print(f"  After F2 (model subset): {df_models['patient_id'].nunique()} patients")

    # ── Mania-contrast subset ─────────────────────────────────────
    def _has_mania_contrast(grp):
        states = set(grp["mood_state"].astype(str).unique())
        return "mania" in states and len(states - {"mania"}) > 0

    mania_contrast_ids = [
        pid for pid, grp in df_models.groupby("patient_id")
        if _has_mania_contrast(grp)
    ]
    df_mania = df_models[df_models["patient_id"].isin(mania_contrast_ids)].copy()
    print(f"  Mania-contrast subset: {df_mania['patient_id'].nunique()} patients "
          f"({mania_contrast_ids})")

    return df_all, df_models, df_mania, available_all



# ═══════════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════════

def _fit(formula: str, data: pd.DataFrame):
    """Fit a patient-grouped mixed-effects model (REML) for the given formula,
    trying each optimizer in _OPTIMIZERS until one converges. Re-raises the last
    exception if none succeed."""
    last = None
    for opt in _OPTIMIZERS:
        try:
            result = mixedlm(formula, data,
                             groups=data["patient_id"]).fit(
                                 reml=True, method=opt, disp=False)
            return result
        except Exception as e:
            last = e
    raise last


def mood_effects(df: pd.DataFrame, feature: str) -> list[dict]:
    """
    Within-person mood effects relative to euthymia, three nested models:
      unadj    : mood term + random intercept
      lang_adj : + language (where language varies)
      full_adj : + word_count_z
    """
    fz  = feature + "_z"
    use = df.dropna(subset=["patient_id", "mood_state", fz]).copy()
    if (len(use) < MIN_OBS_FOR_MODEL
            or use["mood_state"].nunique() < 2
            or use["patient_id"].nunique() < 3):
        return []

    has_lang = "language" in use.columns and use["language"].nunique() > 1
    has_wc   = "word_count_z" in use.columns

    formula_unadj = fz + " ~ " + MOOD_TERM
    formula_lang  = (formula_unadj + " + C(language)") if has_lang else formula_unadj
    formula_full  = (formula_lang + " + word_count_z") if has_wc else formula_lang

    tier_formulas = {
        "unadj":    formula_unadj,
        "lang_adj": formula_lang,
        "full_adj": formula_full,
    }

    rows = []
    for lvl in ("depression", "mania"):
        row = {
            "feature":    feature,
            "contrast":   f"{lvl}_vs_euthymia",
            "n_obs":      len(use),
            "n_patients": use["patient_id"].nunique(),
        }
        for label, formula in tier_formulas.items():
            try:
                r   = _fit(formula, use)
                ci  = r.conf_int()
                key = next((k for k in r.fe_params.index if f"T.{lvl}]" in k), None)
                if key is None:
                    continue
                beta = float(r.fe_params[key])
                row[f"beta_{label}"]      = round(beta, 4)
                row[f"ci_lo_{label}"]     = round(float(ci.loc[key, 0]), 4)
                row[f"ci_hi_{label}"]     = round(float(ci.loc[key, 1]), 4)
                row[f"p_{label}"]         = round(float(r.pvalues[key]), 4)
                row[f"direction_{label}"] = (f"{lvl}>euthymia" if beta > 0
                                             else f"euthymia>{lvl}")
            except Exception:
                pass
        rows.append(row)
    return rows


def mania_subset_effects(df_mania: pd.DataFrame, feature: str) -> dict | None:
    """
    Mania vs non-mania contrast in the mania-contrast subset.
    Binary mood coding (mania=1, non-mania=0). Language-adjusted where it varies.
    NO FDR — purely descriptive / directional consistency check.
    """
    fz  = feature + "_z"
    use = df_mania.dropna(subset=["patient_id", "mood_state", fz]).copy()
    if len(use) < MIN_OBS_FOR_MODEL or use["patient_id"].nunique() < 3:
        return None

    use["is_mania"] = (use["mood_state"].astype(str) == "mania").astype(int)

    terms = ["is_mania"]
    if "language" in use.columns and use["language"].nunique() > 1:
        terms.append("C(language)")
    formula = fz + " ~ " + " + ".join(terms)

    row = {
        "feature":          feature,
        "n_patients":       use["patient_id"].nunique(),
        "n_obs":            len(use),
        "n_mania_obs":      int((use["is_mania"] == 1).sum()),
        "n_nonmania_obs":   int((use["is_mania"] == 0).sum()),
    }
    try:
        r    = _fit(formula, use)
        ci   = r.conf_int()
        beta = float(r.fe_params["is_mania"])
        row.update(
            beta_mania_vs_nonmania = round(beta, 4),
            ci_lo                  = round(float(ci.loc["is_mania", 0]), 4),
            ci_hi                  = round(float(ci.loc["is_mania", 1]), 4),
            p_mania_vs_nonmania    = round(float(r.pvalues["is_mania"]), 4),
            direction              = ("mania>non_mania" if beta > 0
                                      else "non_mania>mania"),
        )
    except Exception:
        row.update(beta_mania_vs_nonmania=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                   p_mania_vs_nonmania=np.nan, direction=np.nan)
    return row


def compute_icc(df: pd.DataFrame, feature: str) -> dict:
    col = feature
    if col not in df.columns:
        return {"feature": feature, "icc": np.nan, "var_between": np.nan,
                "var_within": np.nan, "n_obs": 0, "n_patients": 0}
    data = df[["patient_id", col]].dropna()
    res  = {"feature": feature, "icc": np.nan, "var_between": np.nan,
            "var_within": np.nan, "n_obs": len(data),
            "n_patients": data["patient_id"].nunique()}
    if data["patient_id"].nunique() < 3 or len(data) < 6:
        return res
    try:
        r  = _fit(f"{col} ~ 1", data)
        vb = float(r.cov_re.iloc[0, 0])
        vw = float(r.scale)
        if vb + vw > 0:
            res.update(icc=round(vb / (vb + vw), 4),
                       var_between=round(vb, 4),
                       var_within=round(vw, 4))
    except Exception:
        pass
    return res


def severity_effects(df: pd.DataFrame, feature: str) -> list[dict]:
    """
    Within-person YMRS / HDRS associations, two nested models:
      unadj    : severity term + random intercept
      lang_adj : + language (where language varies)
    Severity scores are person-mean centered (_wp).
    """
    fz  = feature + "_z"
    use = df.dropna(subset=["patient_id", fz]).copy()
    if len(use) < MIN_OBS_FOR_MODEL or use["patient_id"].nunique() < 3:
        return []

    has_lang = "language" in use.columns and use["language"].nunique() > 1

    rows = []
    for score in ("ymrs", "hdrs"):
        swp = score + "_wp"
        if swp not in use.columns:
            print(f"  ⚠ {swp} not found — severity_effects skipped for {score}.")
            continue
        sub = use.dropna(subset=[swp])
        if len(sub) < MIN_OBS_FOR_MODEL:
            continue

        formula_unadj = fz + " ~ " + swp
        formula_lang  = (formula_unadj + " + C(language)") if has_lang else formula_unadj

        tier_formulas = {"unadj": formula_unadj, "lang_adj": formula_lang}

        row = {
            "feature":    feature,
            "score":      score,
            "n_obs":      len(sub),
            "n_patients": sub["patient_id"].nunique(),
        }
        for label, formula in tier_formulas.items():
            try:
                r    = _fit(formula, sub)
                ci   = r.conf_int()
                beta = float(r.fe_params[swp])
                row[f"beta_{label}"]      = round(beta, 4)
                row[f"ci_lo_{label}"]     = round(float(ci.loc[swp, 0]), 4)
                row[f"ci_hi_{label}"]     = round(float(ci.loc[swp, 1]), 4)
                row[f"p_{label}"]         = round(float(r.pvalues[swp]), 4)
                row[f"direction_{label}"] = ("feature>severity" if beta > 0
                                             else "feature<severity")
            except Exception:
                pass
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════
# TRAJECTORY CSV  (for downstream plotting)
# ═══════════════════════════════════════════════════════════════════

def _patient_group(states: set[str]) -> str:
    """Classify a patient by which mood states they contribute (e.g.
    'all_three_states', 'mania_depression_only') for grouping in the trajectory
    CSV."""
    has_mania = "mania"      in states
    has_dep   = "depression" in states
    has_euth  = "euthymia"   in states
    if has_mania and has_dep and has_euth:
        return "all_three_states"
    if has_mania and has_dep and not has_euth:
        return "mania_depression_only"
    if has_mania and has_euth and not has_dep:
        return "mania_euthymia_only"
    if has_dep and has_euth and not has_mania:
        return "depression_euthymia_only"
    return "other:" + "_".join(sorted(states))


def build_trajectory_csv(df_models: pd.DataFrame,
                         available: list[str]) -> pd.DataFrame:
    """Build the per-visit long-format trajectory table used by the visualisation
    layer: one row per patient-visit with mood state, available contrasts, and
    both raw and z-scored feature values, ordered with T1 metrics first."""
    rows = []
    for pat_id, grp in df_models.groupby("patient_id"):
        if "visit" in grp.columns:
            grp = grp.sort_values("visit").reset_index(drop=True)
        else:
            grp = grp.reset_index(drop=True)

        actual_states = set(grp["mood_state"].astype(str).unique())
        group_label   = _patient_group(actual_states)

        contrasts = []
        if "mania"      in actual_states and "depression" in actual_states:
            contrasts.append("mania_vs_depression")
        if "mania"      in actual_states and "euthymia"   in actual_states:
            contrasts.append("mania_vs_euthymia")
        if "depression" in actual_states and "euthymia"   in actual_states:
            contrasts.append("depression_vs_euthymia")
        contrasts_str = "|".join(contrasts) if contrasts else "none"

        for rank, (_, row) in enumerate(grp.iterrows(), start=1):
            entry = {
                "patient_id":          pat_id,
                "patient_group":       group_label,
                "contrasts_available": contrasts_str,
                "visit_rank":          rank,
                "visit":               row.get("visit", rank),
                "mood_state":          str(row["mood_state"]),
                "episode_id":          row.get("episode_id", ""),
            }
            for m in available:
                entry[m]        = row.get(m,        np.nan)
                entry[m + "_z"] = row.get(m + "_z", np.nan)
            rows.append(entry)

    traj_df = pd.DataFrame(rows)

    metric_cols = (
        [m for m in T1_METRICS if m in available] +
        [m for m in available  if m not in T1_METRICS]
    )
    col_order = (
        ["patient_id", "patient_group", "contrasts_available",
         "visit_rank", "visit", "mood_state", "episode_id"] +
        [c for m in metric_cols for c in (m, m + "_z") if c in traj_df.columns]
    )
    traj_df = traj_df[[c for c in col_order if c in traj_df.columns]]

    summary = (traj_df.drop_duplicates("patient_id")
               .groupby("patient_group")["patient_id"]
               .apply(lambda x: sorted(x.tolist(), key=int)))
    print("  Trajectory groups:")
    for grp_name, pats in summary.items():
        print(f"    {grp_name}: n={len(pats)}  {pats}")

    return traj_df


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_longitudinal_analysis() -> None:
    """Run the full RQ3 longitudinal pipeline: load and filter the sessions data,
    fit the mood-effect, ICC, severity and mania-subset models, build the
    trajectory table, write all output CSVs to OUTPUT_DIR, and print a console
    summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LONGITUDINAL ANALYSIS")
    print("Methodology: Speech Graphs  |  Task: Emotion")
    print("Severity predictors: within-person centered (_wp)")
    print("Graph features: population z-scored (_z)")
    print("Model tiers: unadjusted / language-adjusted / full (+word count)")
    print("=" * 70)

    df_all, df_models, df_mania, available = load_data()

    if not available:
        print("✗ No metrics with non-zero variance — check data.")
        return

    print(f"\n  Metrics: {available}")

    # ── A: Mood effects ───────────────────────────────────────────
    print("\n  [A] Fitting mood-effects models (unadj / lang / full)...")
    mood_rows = [r for f in available for r in mood_effects(df_models, f)]
    if not mood_rows:
        print("✗ No mood-effect rows — check data / MIN_OBS_FOR_MODEL.")
        return

    mood_df = pd.DataFrame(mood_rows)
    for adj in MOOD_TIERS:
        pcol, fdrcol = f"p_{adj}", f"p_{adj}_fdr"
        mood_df[fdrcol] = np.nan
        if pcol not in mood_df.columns:
            continue
        valid = mood_df[pcol].notna()
        if valid.sum() > 0:
            _, q, _, _ = multipletests(mood_df.loc[valid, pcol], method="fdr_bh")
            mood_df.loc[valid, fdrcol] = q.round(4)

    if "p_unadj_fdr" in mood_df.columns:
        mood_df["sig_unadj"] = mood_df["p_unadj_fdr"] < FDR_ALPHA
    if "beta_unadj" in mood_df.columns:
        mood_df["meets_effect_floor"] = mood_df["beta_unadj"].abs() >= BETA_FLOOR
    sort_col = "p_unadj" if "p_unadj" in mood_df.columns else "p_lang_adj"
    mood_df = mood_df.sort_values(sort_col).reset_index(drop=True)

    # ── B: ICC ────────────────────────────────────────────────────
    print("\n  [B] Computing ICC (F1 sample, raw values, unconditional)...")
    icc_rows = [compute_icc(df_all, f) for f in available]
    icc_df   = pd.DataFrame(icc_rows)
    icc_df["trait_like"] = icc_df["icc"] >= 0.5
    icc_df = icc_df.sort_values("icc", ascending=False,
                                na_position="last").reset_index(drop=True)

    # ── C: Severity effects ───────────────────────────────────────
    print("\n  [C] Fitting severity models (unadj / lang, within-person centered)...")
    sev_rows = [r for f in available for r in severity_effects(df_models, f)]
    sev_df   = pd.DataFrame(sev_rows) if sev_rows else pd.DataFrame()
    if not sev_df.empty:
        for adj in SEV_TIERS:
            pcol, fdrcol = f"p_{adj}", f"p_{adj}_fdr"
            sev_df[fdrcol] = np.nan
            if pcol not in sev_df.columns:
                continue
            valid = sev_df[pcol].notna()
            if valid.sum() > 0:
                _, q, _, _ = multipletests(sev_df.loc[valid, pcol], method="fdr_bh")
                sev_df.loc[valid, fdrcol] = q.round(4)
        if "p_unadj_fdr" in sev_df.columns:
            sev_df["sig_unadj"] = sev_df["p_unadj_fdr"] < FDR_ALPHA
        if "beta_unadj" in sev_df.columns:
            sev_df["meets_effect_floor"] = sev_df["beta_unadj"].abs() >= BETA_FLOOR
        sort_col = "p_unadj" if "p_unadj" in sev_df.columns else "p_lang_adj"
        sev_df = sev_df.sort_values(sort_col).reset_index(drop=True)

    # ── D: Mania-subset sub-analysis ─────────────────────────────
    print(f"\n  [D] Mania-subset sub-analysis "
          f"(n={df_mania['patient_id'].nunique()} patients, descriptive only)...")
    mania_rows = [r for f in available
                  if (r := mania_subset_effects(df_mania, f)) is not None]
    mania_df   = pd.DataFrame(mania_rows) if mania_rows else pd.DataFrame()
    if not mania_df.empty:
        mania_df["meets_effect_floor"] = (
            mania_df["beta_mania_vs_nonmania"].abs() >= BETA_FLOOR)
        mania_df = mania_df.sort_values("p_mania_vs_nonmania").reset_index(drop=True)

    # ── E: Trajectory CSV ─────────────────────────────────────────
    print("\n  [E] Building trajectory CSV...")
    traj_df = build_trajectory_csv(df_models, available)

    # ── Save outputs ──────────────────────────────────────────────
    mood_df.to_csv(OUTPUT_DIR / "table_A_mood_effects.csv",        index=False)
    icc_df.to_csv( OUTPUT_DIR / "table_B_icc.csv",                 index=False)
    if not sev_df.empty:
        sev_df.to_csv(OUTPUT_DIR / "table_C_severity_effects.csv", index=False)
    if not mania_df.empty:
        mania_df.to_csv(OUTPUT_DIR / "table_D_mania_subset.csv",   index=False)
    traj_df.to_csv(OUTPUT_DIR / "table_E_trajectories.csv",        index=False)

    # ── Console summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n[A] Mood effects — betas across tiers (sorted by p_unadj):")
    show_cols = ["feature", "contrast"]
    for adj in MOOD_TIERS:
        for c in (f"beta_{adj}", f"p_{adj}"):
            if c in mood_df.columns:
                show_cols.append(c)
    print(mood_df[[c for c in show_cols if c in mood_df.columns]].to_string(index=False))

    print("\n[A] FDR-significant mood effects (any tier):")
    sig_any = pd.Series(False, index=mood_df.index)
    for adj in MOOD_TIERS:
        col = f"p_{adj}_fdr"
        if col in mood_df.columns:
            sig_any |= mood_df[col] < FDR_ALPHA
    if not sig_any.any():
        print("  none")
    else:
        print(mood_df[sig_any][[c for c in show_cols if c in mood_df.columns]]
              .to_string(index=False))

    print("\n[B] ICC summary (sorted by ICC descending):")
    print(icc_df[["feature", "icc", "var_between", "var_within",
                  "trait_like"]].to_string(index=False))

    print("\n[C] Severity effects — betas across tiers:")
    if sev_df.empty:
        print("  none")
    else:
        sev_cols = ["feature", "score"]
        for adj in SEV_TIERS:
            for c in (f"beta_{adj}", f"p_{adj}"):
                if c in sev_df.columns:
                    sev_cols.append(c)
        print(sev_df[[c for c in sev_cols if c in sev_df.columns]].to_string(index=False))

        print("\n[C] Nominally significant severity (p_unadj < 0.05):")
        if "p_unadj" in sev_df.columns:
            nom_sev = sev_df[sev_df["p_unadj"] < 0.05]
            print("  none" if nom_sev.empty
                  else nom_sev[[c for c in sev_cols if c in sev_df.columns]]
                  .to_string(index=False))

    print(f"\n[D] Mania-subset (n={df_mania['patient_id'].nunique()} patients, "
          f"DESCRIPTIVE ONLY — no FDR):")
    if mania_df.empty:
        print("  none")
    else:
        print(mania_df[["feature", "beta_mania_vs_nonmania", "ci_lo", "ci_hi",
                        "p_mania_vs_nonmania", "direction",
                        "meets_effect_floor"]].to_string(index=False))

    print(f"\n[E] Trajectory CSV: {len(traj_df)} rows, "
          f"{traj_df['patient_id'].nunique()} patients, "
          f"groups: {traj_df['patient_group'].value_counts().to_dict()}")

    print(f"\nResults saved → {OUTPUT_DIR}/")
    for fname in ["table_A_mood_effects.csv", "table_B_icc.csv",
                  "table_C_severity_effects.csv", "table_D_mania_subset.csv",
                  "table_E_trajectories.csv"]:
        exists = (OUTPUT_DIR / fname).exists()
        print(f"  {'✓' if exists else '✗'} {fname}")


if __name__ == "__main__":
    run_longitudinal_analysis()
