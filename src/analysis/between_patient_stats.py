"""
Curated between-patient statistical analysis.

For each feature family, analyses mood-state group differences and YMRS/HDRS
severity correlations per task (emotion / clinical_evaluation / bdae), reported
separately with FDR scoped within (family × task). Each family has one a-priori
primary config; other configs contribute robustness (rank-order consistency with
the primary). Findings are assigned to evidence tiers (T1 language+word-count
adjusted, T2 language-only, T3 unadjusted, T4 effect-only) subject to an effect
floor of eps2 >= 0.06 OR max|r_rb| >= 0.30; FDR is displayed but never gates
tiers. Outputs are written to results/between_patient_curated/ (primary stats,
tiered group differences, dimensional severity, word-count baseline, robustness
raw, feature reference, and a JSON summary).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import RESULTS_DIR
from src.stats_helpers import (
    analyze_experiment,
    classify_pattern,
    consistency_across_experiments,
    fold_mixed,
)

_RES = Path(RESULTS_DIR)
OUT  = _RES / "between_patient_curated"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TASKS            = ["emotion", "clinical_evaluation", "bdae"]
EMBEDDING_TASKS  = ["emotion", "clinical_evaluation"]
EMBEDDING_FAMILIES = {"semantic_networks", "semantic_similarity", "topic"}
TOPIC_MODELS     = ["mpnet", "minilm", "e5_large", "labse"]
SENTENCE_MODELS  = ["mpnet", "minilm", "e5_large", "labse"]

PRIMARY = {
    "speech_graphs":        {"experiment_contains": "w10s1"},
    "cooccurrence_emotion": {"window": 3},
    "semantic_networks":    {"source_startswith": "xlm", "threshold": 0.50,
                             "stopwords_removed": False},
    "semantic_similarity":  {"source_startswith": "xlm", "stopwords_removed": False},
}

P_THRESH    = 0.05
EPS2_THRESH = 0.06
RRB_THRESH  = 0.30


def _tasks_for(fam: str) -> list[str]:
    """Return the list of tasks a family is analysed over. Embedding-based
    families (semantic networks, semantic similarity, topic) only have emotion
    and clinical_evaluation transcripts, so they use EMBEDDING_TASKS; all other
    families use the full TASKS list including bdae."""
    return EMBEDDING_TASKS if (fam in EMBEDDING_FAMILIES or fam.startswith("topic")) else TASKS


def _collapse_family(fam: str) -> str:
    """Collapse per-model family names into their canonical family so that
    robustness configs can be matched to their primary row. Topic model variants
    map to "topic" and per-model sentence-similarity variants map to
    "semantic_similarity_sentence"; everything else is returned unchanged."""
    if fam.startswith("topic_"): return "topic"
    if fam.startswith("semantic_similarity_sent_"): return "semantic_similarity_sentence"
    return fam


def _sf(x, default=float("nan")):
    """Safely coerce a value to float, returning the given default (NaN by
    default) if the conversion fails."""
    try:    return float(x)
    except: return default


# ═══════════════════════════════════════════════════════════════════════════════
# CURATED KEEP-LISTS
# ═══════════════════════════════════════════════════════════════════════════════

KEEP_LISTS: dict[str, dict[str, str]] = {
    "lexical": {
        "type_token_ratio": "Lexical diversity. Expect depression<euthymia; length-confounded.",
        "mean_word_length":  "Surface complexity. Weak prior.",
        "repetition_rate":   "Consecutive same-word bigrams. Expect elevated in depression/mania.",
    },
    "psycholinguistic": {
        "z_anger":        "Negative emotion. Expect depression-elevated.",
        "z_disgust":      "Negative emotion. Expect depression-elevated.",
        "z_fear":         "Negative emotion. Expect depression-elevated.",
        "z_sadness":      "Clearest depressive marker. depression>euthymia>mania.",
        "z_surprise":     "Negative emotion. Expect depression-elevated.",
        "z_joy":          "Positive/approach. Expect mania>euthymia>depression.",
        "z_anticipation": "Approach/future-oriented. Expect mania-elevated.",
        "z_trust":        "Positive social. Expect mania-elevated, depression-reduced.",
        "mean_valence":   "Overall affective tone. Expect mania>euthymia>depression.",
        "mean_arousal":   "Affective activation. Expect mania elevated, depression reduced.",
        "pct_positive_words": "Composition. Expect mania-elevated.",
        "pct_negative_words": "Composition. Expect depression-elevated.",
        "emotional_entropy":  "Valence-transition variability. Exploratory.",
        "first_person_rate":  "Self-reference. Strong prior: depression>euthymia>mania.",
        "filler_rate":        "Disfluency. Expect elevated with psychomotor slowing.",
    },
    "speech_graphs": {
     #   "Nodes":    "Vocabulary/graph size. Verbosity-adjacent.",
     #   "Edges":    "Total edges. Verbosity-adjacent.",
        "LSC":      "Largest strongly connected component. Expect mania-elevated.",
     #   "ATD":      "Average total degree. Expect mania higher.",
        "Density":  "Connectivity normalised by size.",
        "ASP":      "Average shortest path. Expect mania shorter.",
   #     "Diameter": "Longest shortest path.",
        "CC":       "Clustering / local cohesion.",
        "RE":       "Repeated edges (recurrence).",
    #    "PE":       "Parallel edges (recurrence motif).",
        "L1":       "Self-loops / immediate repetition.",
        "L2":       "Two-step cycles.",
     #   "L3":       "Three-step cycles.",
        "LSC_z":    "Structure vs random baseline.",
    },
    "cooccurrence_node": {
        "pct_positive_nodes": "Affective composition. Expect mania more positive.",
        "pct_negative_nodes": "Affective composition. Expect depression more negative.",
        "density":            "Connectivity.",
        "mean_shortest_path": "Average path length.",
        "diameter":           "Longest shortest path.",
        "mean_clustering":    "Local cohesion.",
        "modularity":         "Community structure.",
        "valence_homophily":  "Same-valence word connectivity.",
    },
    "cooccurrence_triad": {
        "valence_dob_z":                  "Structural balance (valence), z-scored. FRAGILE on short transcripts.",
        "arousal_dob_z":                  "Structural balance (arousal), z-scored. FRAGILE.",
        "dominance_dob_z":                "Structural balance (dominance), z-scored. FRAGILE.",
        "emotional_complexity_z_shuffled":"Emotional complexity z-scored. FRAGILE.",
    },
    "semantic_networks": {
        "mean_degree":        "Average node degree.",
        "mean_clustering":    "Clustering coefficient.",
        "diameter":           "Network diameter.",
        "mean_shortest_path": "Average shortest path.",
        "density":            "Connectivity.",
        "modularity":         "Community structure.",
        "mean_edge_weight":   "Mean semantic similarity. Expect lower in symptomatic states.",
    },
    "semantic_similarity": {
        "sem_mean": "Mean consecutive similarity. Expect reduced in mania.",
        "sem_std":  "Coherence variability. Expect elevated in disorganised speech.",
        "sem_min":  "Lowest coherence point.",
        "sem_max":  "Highest coherence point.",
    },
    "topic": {
        "topic_switch_rate":   "Consecutive topic change. Expect mania higher.",
        "pct_symptom_windows": "Illness meta-language rate. Exploratory.",
    },
}

FAMILY_SOURCES = {
    "lexical": {
        "path": _RES / "lexical" / "lexical_patient_features.csv",
        "keep": "lexical", "primary": None,
    },
    "psycholinguistic": {
        "path": _RES / "psycholinguistic" / "psycholinguistic_patient_features.csv",
        "keep": "psycholinguistic", "primary": None,
    },
    "speech_graphs": {
        "path": _RES / "speech_graphs" / "speech_graphs_patient_values.csv",
        "keep": "speech_graphs", "primary": "speech_graphs",
    },
    "semantic_networks": {
        "path": _RES / "semantic_networks" / "semantic_networks_patient_features.csv",
        "keep": "semantic_networks", "primary": "semantic_networks",
    },
}

COOC_PATH   = _RES / "cooccurrence_emotion" / "cooccurrence_emotion_patient_values.csv"
SEMSIM_PATH = _RES / "semantic_similarity"  / "semantic_patient_features.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_state(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the symptom-state column name, renaming a legacy
    'episode_type' column to 'symptom_state' when the latter is absent."""
    if "symptom_state" not in df.columns and "episode_type" in df.columns:
        df = df.rename(columns={"episode_type": "symptom_state"})
    return df


def _apply_primary(df: pd.DataFrame, fam: str, task: str) -> pd.DataFrame:
    """Filter a family's raw feature frame to its a-priori primary config for a
    task: applies the task filter plus the family's selectors in PRIMARY
    (experiment substring, window, embedding source/threshold, stopword flag)."""
    out = df.copy()
    tcol = next((c for c in ("task", "task_set") if c in out.columns), None)
    if tcol:
        out = out[out[tcol].astype(str) == task]
    spec = PRIMARY.get(fam, {})
    if "experiment_contains" in spec and "experiment" in out.columns:
        out = out[out["experiment"].astype(str).str.contains(spec["experiment_contains"])]
    if "window" in spec and "window" in out.columns:
        out = out[out["window"] == spec["window"]]
    if "source_startswith" in spec and "source" in out.columns:
        out = out[out["source"].astype(str).str.startswith(spec["source_startswith"])]
    if "threshold" in spec and "threshold" in out.columns:
        out = out[np.isclose(out["threshold"].astype(float), spec["threshold"])]
    if "stopwords_removed" in spec and "stopwords_removed" in out.columns:
        out = out[out["stopwords_removed"] == spec["stopwords_removed"]]
    return out

def _feature_cols(df: pd.DataFrame, keep: dict[str, str],
                  tag: str = "") -> tuple[list[str], dict[str, str]]:
    """
    Returns (usable_features, exclusion_reasons).
    exclusion_reasons maps feature_name → reason string for excluded features.
    """
    usable   = []
    excluded = {}
    for f in keep:
        if f not in df.columns:
            reason = "not-in-columns"
            print(f"      ⚠ {tag}EXCLUDED '{f}': {reason}"); excluded[f] = reason; continue
        col = pd.to_numeric(df[f], errors="coerce")
        nn  = col.notna().sum()
        if nn == 0:
            reason = "all-NaN"
            print(f"      ⚠ {tag}EXCLUDED '{f}': {reason}"); excluded[f] = reason; continue
        if col.nunique(dropna=True) <= 1:
            reason = f"zero-variance (n={nn})"
            print(f"      ⚠ {tag}EXCLUDED '{f}': {reason}"); excluded[f] = reason; continue
        df[f] = col
        usable.append(f)
    return usable, excluded


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a feature frame for analysis: normalise the state column, lower-case
    caliber codes, fold mixed episodes, restrict to the three analysed mood states
    (mania/euthymia/depression), and deduplicate to one row per patient."""
    df = _norm_state(df).copy()
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
    df = fold_mixed(df)
    df = df[df["symptom_state"].isin(["mania", "euthymia", "depression"])]
    return df.drop_duplicates("caliber_code")


def _freeze_features(primary_by_task: dict[str, pd.DataFrame],
                     keep: dict[str, str],
                     fam: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """
    Returns (frozen_feature_list, per_task_exclusions).
    per_task_exclusions: {task: {feature: reason}} for features excluded in that task.
    A feature is frozen (kept) if it passes _feature_cols in ANY task's pooled data.
    Per-task exclusions are recorded independently for the reference table.
    """
    if not primary_by_task:
        return [], {}

    # Per-task exclusion audit
    per_task_exclusions: dict[str, dict[str, str]] = {}
    feature_passed_any: set[str] = set()

    for task, df in primary_by_task.items():
        usable, excl = _feature_cols(df, keep, f"[{fam}|{task}] ")
        per_task_exclusions[task] = excl
        feature_passed_any.update(usable)

    # Freeze = passed in at least one task (consistent with old pooled logic)
    pooled = pd.concat(list(primary_by_task.values()), ignore_index=True)
    frozen, _ = _feature_cols(pooled, {f: keep[f] for f in feature_passed_any}, f"[{fam}|POOLED] ")

    if frozen:
        print(f"    [{fam}] frozen features ({len(frozen)}): {frozen}")
    return frozen, per_task_exclusions

def _build_feature_rows(fam: str, keep: dict[str, str],
                        frozen: list[str],
                        per_task_excl: dict[str, dict[str, str]],
                        tasks_run: list[str]) -> list[dict]:
    """
    One row per (family × feature × task).
    kept=True  → feature passed and was used in that task's analysis.
    kept=False → feature was excluded; exclusion_reason explains why.
    kept=None  → task not run for this family (e.g. bdae for embedding families).
    """
    rows = []
    for f, interp in keep.items():
        for task in tasks_run:
            if task not in per_task_excl:
                # Task not attempted for this family
                rows.append({"family": fam, "feature": f, "task": task,
                             "kept": None, "exclusion_reason": "task-not-run",
                             "interpretation": interp})
            elif f in per_task_excl[task]:
                rows.append({"family": fam, "feature": f, "task": task,
                             "kept": False,
                             "exclusion_reason": per_task_excl[task][f],
                             "interpretation": interp})
            else:
                rows.append({"family": fam, "feature": f, "task": task,
                             "kept": f in frozen,
                             "exclusion_reason": "" if f in frozen else "not-frozen-pooled",
                             "interpretation": interp})
    return rows


def _run_primary(df: pd.DataFrame, feats: list[str],
                 fam: str, task: str) -> pd.DataFrame | None:
    """Run the primary group-difference analysis for one family × task on the
    given frozen feature list. Prepares the data, requires at least 10 patients,
    calls analyze_experiment, and tags the result with family, task, a
    '<fam>_<task>_primary' experiment label, and the source experiment. Returns
    None when there are too few patients or no usable features/results."""
    feats = [f for f in feats if f in df.columns]
    if not feats:
        return None
    src = None
    if "experiment" in df.columns:
        m = df["experiment"].mode()
        src = str(m.iloc[0]) if not m.empty else None
    df = _prep(df)
    if len(df) < 10:
        print(f"    [{fam}|{task}] n={len(df)} <10 — skip"); return None
    print(f"    [{fam}|{task}] n={len(df)}  features={feats}")
    res = analyze_experiment(df, feats)
    if res is None or res.empty:
        return None
    res.insert(0, "family", fam)
    res.insert(1, "task",   task)
    res.insert(2, "experiment", f"{fam}_{task}_primary")
    res["source_experiment"] = src
    return res


def _run_robustness(df_all: pd.DataFrame, keep: dict[str, str],
                    fam: str, task: str,
                    config_col: str) -> pd.DataFrame | None:
    """
    Run analyze_experiment for each nuisance config and return a long frame
    with one row per (config × metric). Caller uses this to count how many
    configs reproduce the primary experiment's rank order.
    No consistency summary is computed here — that happens at export time.
    """
    tcol = next((c for c in ("task", "task_set") if c in df_all.columns), None)
    sub  = df_all[df_all[tcol].astype(str) == task] if tcol else df_all
    if config_col not in sub.columns:
        return None
    rows = []
    for cfg, g in sub.groupby(config_col):
        feats, _ = _feature_cols(g, keep)
        if not feats:
            continue
        g2 = _prep(g)
        if len(g2) < 10:
            continue
        r = analyze_experiment(g2, feats)
        if r is None or r.empty:
            continue
        r["config"] = str(cfg)
        rows.append(r)
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    long.insert(0, "family", fam)
    long.insert(1, "task",   task)
    return long


# ═══════════════════════════════════════════════════════════════════════════════
# WORD COUNT BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _wordcount_analysis(per_task_csv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute the word-count verbosity baseline per task: an unadjusted
    Kruskal-Wallis omnibus across mood states with ε², pairwise rank-biserial
    effect sizes, a language-adjusted (rank-ANCOVA) group p-value, and YMRS/HDRS
    severity Spearman correlations at three adjustment levels. FDR is applied
    across tasks within each adjusted p-column. Returns one row per task."""
    from scipy.stats import kruskal, mannwhitneyu, spearmanr
    from statsmodels.stats.multitest import multipletests

    ORDER = ["mania", "euthymia", "depression"]
    SCALES = {
        "ymrs": ["ymrs", "ymrs_total", "YMRS", "YMRS_total"],
        "hdrs": ["hdrs", "hdrs_total", "HDRS", "HDRS_total", "ham_d", "hamd"],
    }
    rows = []

    for task, raw in per_task_csv.items():
        df = _prep(raw)
        if "word_count" not in df.columns or len(df) < 10:
            continue
        df = df[df["word_count"].notna()].copy()

        gs = {s: df.loc[df["symptom_state"] == s, "word_count"].values for s in ORDER}
        gs = {s: v for s, v in gs.items() if len(v) > 0}
        if len(gs) < 2:
            continue

        # ── Unadjusted omnibus (Kruskal-Wallis) ─────────────────────────────
        H, p_kw = kruskal(*gs.values())
        n, k    = sum(len(v) for v in gs.values()), len(gs)
        meds    = {s: float(np.median(v)) for s, v in gs.items()}

        row = {
            "feature":        "word_count",
            "task":           task,
            "rank_order":     ">".join(sorted(meds, key=meds.get, reverse=True)),
            "med_mania":      meds.get("mania"),
            "med_euthymia":   meds.get("euthymia"),
            "med_depression": meds.get("depression"),
            "p_kw":           float(p_kw),
            "epsilon2":       (H - k + 1) / (n - k) if n > k else np.nan,
            "p_lang_adj":     np.nan,
            "p_lang_adj_fdr": np.nan,
        }

        # ── Pairwise rank-biserial effect sizes ──────────────────────────────
        for s1, s2, lab in [("mania", "euthymia", "ma_eu"),
                             ("mania", "depression", "ma_de"),
                             ("euthymia", "depression", "eu_de")]:
            if s1 in gs and s2 in gs:
                U, pp = mannwhitneyu(gs[s1], gs[s2], alternative="two-sided")
                row[f"r_rb_{lab}"] = 1 - 2 * U / (len(gs[s1]) * len(gs[s2]))
                row[f"p_mw_{lab}"] = float(pp)

        # ── Language-adjusted group p (rank-based ANCOVA) ────────────────────
        has_lang = ("language" in df.columns
                    and df["language"].notna().any()
                    and df["language"].nunique() >= 2)
        if has_lang:
            try:
                tmp = df[["word_count", "language", "symptom_state"]].dropna().copy()
                tmp = tmp[tmp["symptom_state"].isin(ORDER)]
                if len(tmp) >= 10 and tmp["symptom_state"].nunique() >= 2:
                    tmp["_rank"] = tmp["word_count"].rank(method="average")
                    lang_dummies = pd.get_dummies(tmp["language"], drop_first=True,
                                                  dtype=float)
                    if lang_dummies.shape[1] > 0:
                        from numpy.linalg import lstsq
                        X = np.column_stack([np.ones(len(tmp)), lang_dummies.values])
                        y = tmp["_rank"].values
                        coef, *_ = lstsq(X, y, rcond=None)
                        tmp["_resid"] = y - X @ coef
                    else:
                        tmp["_resid"] = tmp["_rank"]
                    resid_gs = {s: tmp.loc[tmp["symptom_state"] == s, "_resid"].values
                                for s in ORDER if (tmp["symptom_state"] == s).any()}
                    if len(resid_gs) >= 2:
                        _, p_adj = kruskal(*resid_gs.values())
                        row["p_lang_adj"] = float(p_adj)
            except Exception:
                pass

        # ── Clinical severity correlations (Spearman) ────────────────────────
        # Three tiers mirroring analyze_experiment(): unadjusted, language-adjusted
        # (partial), and full (language + word-count). Since this feature IS
        # word_count, full_adj == lang_adj; both are populated for schema
        # consistency with the rest of the pipeline.
        for scale, candidates in SCALES.items():
            scale_col = next((c for c in candidates if c in df.columns), None)
            if scale_col is None:
                for key in (f"r_{scale}", f"p_{scale}",
                            f"r_{scale}_lang_adj", f"p_{scale}_lang_adj",
                            f"r_{scale}_full_adj", f"p_{scale}_full_adj"):
                    row[key] = np.nan
                continue

            # ── Unadjusted ───────────────────────────────────────────────────
            pair = df[["word_count", scale_col]].dropna()
            if len(pair) >= 6:
                r_raw, p_raw = spearmanr(pair["word_count"], pair[scale_col])
                row[f"r_{scale}"] = float(r_raw)
                row[f"p_{scale}"] = float(p_raw)
            else:
                row[f"r_{scale}"] = np.nan
                row[f"p_{scale}"] = np.nan

            # ── Language-adjusted (residualise both vars on language dummies) ─
            if has_lang:
                try:
                    tmp2 = df[["word_count", scale_col, "language"]].dropna().copy()
                    if len(tmp2) >= 10 and tmp2["language"].nunique() >= 2:
                        lang_d = pd.get_dummies(tmp2["language"], drop_first=True,
                                                dtype=float)
                        if lang_d.shape[1] > 0:
                            from numpy.linalg import lstsq
                            X = np.column_stack([np.ones(len(tmp2)), lang_d.values])
                            for col_name in ["word_count", scale_col]:
                                y = tmp2[col_name].values.astype(float)
                                coef, *_ = lstsq(X, y, rcond=None)
                                tmp2[f"_resid_{col_name}"] = y - X @ coef
                            r_la, p_la = spearmanr(
                                tmp2["_resid_word_count"],
                                tmp2[f"_resid_{scale_col}"]
                            )
                            row[f"r_{scale}_lang_adj"] = float(r_la)
                            row[f"p_{scale}_lang_adj"] = float(p_la)
                            # full_adj == lang_adj for word_count (no self-adjustment)
                            row[f"r_{scale}_full_adj"] = float(r_la)
                            row[f"p_{scale}_full_adj"] = float(p_la)
                        else:
                            row[f"r_{scale}_lang_adj"] = row[f"r_{scale}"]
                            row[f"p_{scale}_lang_adj"] = row[f"p_{scale}"]
                            row[f"r_{scale}_full_adj"] = row[f"r_{scale}"]
                            row[f"p_{scale}_full_adj"] = row[f"p_{scale}"]
                    else:
                        row[f"r_{scale}_lang_adj"] = np.nan
                        row[f"p_{scale}_lang_adj"] = np.nan
                        row[f"r_{scale}_full_adj"] = np.nan
                        row[f"p_{scale}_full_adj"] = np.nan
                except Exception:
                    row[f"r_{scale}_lang_adj"] = np.nan
                    row[f"p_{scale}_lang_adj"] = np.nan
                    row[f"r_{scale}_full_adj"] = np.nan
                    row[f"p_{scale}_full_adj"] = np.nan
            else:
                # No language covariate — adjusted == unadjusted
                row[f"r_{scale}_lang_adj"] = row[f"r_{scale}"]
                row[f"p_{scale}_lang_adj"] = row[f"p_{scale}"]
                row[f"r_{scale}_full_adj"] = row[f"r_{scale}"]
                row[f"p_{scale}_full_adj"] = row[f"p_{scale}"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # ── FDR across tasks within each p-column ────────────────────────────────
    # Mirrors the pipeline convention: FDR displayed beside adjusted p-values.
    for scale in SCALES:
        for suffix in ("_lang_adj", "_full_adj"):
            pcol = f"p_{scale}{suffix}"
            fdrcol = f"p_{scale}{suffix}_fdr"
            if pcol not in out.columns:
                continue
            valid = out[pcol].notna()
            if valid.sum() < 2:
                out[fdrcol] = np.nan
                continue
            _, pvals_fdr, _, _ = multipletests(
                out.loc[valid, pcol], alpha=0.05, method="fdr_bh"
            )
            out[fdrcol] = np.nan
            out.loc[valid, fdrcol] = pvals_fdr

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_sig_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the intermediate 'sig_*' boolean flag columns produced by
    analyze_experiment so they do not clutter the exported tables."""
    return df.drop(columns=[c for c in df.columns if c.startswith("sig_")], errors="ignore")


def _maxabs_rrb(df: pd.DataFrame) -> pd.Series:
    """Return, per row, the largest absolute pairwise rank-biserial effect size
    across the three mood contrasts (mania-euthymia, mania-depression,
    euthymia-depression). Used as one of the two effect-size floor criteria."""
    cols = [c for c in ["r_rb_ma_eu", "r_rb_ma_de", "r_rb_eu_de"] if c in df.columns]
    return df[cols].abs().max(axis=1) if cols else pd.Series(0.0, index=df.index)


# Reader-friendly column names — applied only to exported highlight tables.
RENAME_GROUP = {
    "tier":                "tier",
    "family":              "family",
    "metric":              "feature",
    "rank_order":          "mood_ordering",
    "med_mania":           "med_mania",
    "med_euthymia":        "med_euthymia",
    "med_depression":      "med_depression",
    "epsilon2":            "eps2",
    "max_abs_rrb":         "max_abs_r_rb",
    "r_rb_ma_eu":          "r_rb_ma_eu",
    "r_rb_ma_de":          "r_rb_ma_de",
    "r_rb_eu_de":          "r_rb_eu_de",
    "p_kw":                "p_unadjusted",
    "p_mood_lang_adj":     "p_lang_adj",
    "p_mood_lang_adj_fdr": "p_lang_adj_fdr",
    "p_mood_adj":          "p_full_adj",
    "p_mood_adj_fdr":      "p_full_adj_fdr",
    "r_wc":                "r_wc",
    "order_consistency":   "order_consistency",   # already readable: "X/N"
}

RENAME_DIM = {
    "family":                "family",
    "metric":                "feature",
    "rank_order":            "mood_ordering",
    "r_ymrs":                "r_ymrs",
    "p_ymrs":                "p_ymrs",
    "r_ymrs_lang_adj":       "r_ymrs_lang_adj",
    "p_ymrs_lang_adj":       "p_ymrs_lang_adj",
    "p_ymrs_lang_adj_fdr":   "p_ymrs_lang_adj_fdr",
    "r_ymrs_full_adj":       "r_ymrs_full_adj",
    "p_ymrs_full_adj":       "p_ymrs_full_adj",
    "p_ymrs_full_adj_fdr":   "p_ymrs_full_adj_fdr",
    "r_hdrs":                "r_hdrs",
    "p_hdrs":                "p_hdrs",
    "r_hdrs_lang_adj":       "r_hdrs_lang_adj",
    "p_hdrs_lang_adj":       "p_hdrs_lang_adj",
    "p_hdrs_lang_adj_fdr":   "p_hdrs_lang_adj_fdr",
    "r_hdrs_full_adj":       "r_hdrs_full_adj",
    "p_hdrs_full_adj":       "p_hdrs_full_adj",
    "p_hdrs_full_adj_fdr":   "p_hdrs_full_adj_fdr",
    "r_wc":                  "r_wc",
    "epsilon2":              "eps2",
}

TIER_LABELS = {
    "1_full_robust":   "1_full_robust",
    "2_language_only": "2_language_only",
    "3_unadjusted":    "3_unadjusted",
    "4_effect_only":   "4_effect_only",
}


def _compute_order_consistency(allp: pd.DataFrame,
                                rob_raw: pd.DataFrame | None) -> pd.Series:
    """
    For each primary row, count how many robustness configs share the EXACT
    same rank_order as the primary, expressed as the string "X/N".
    Pinned to the primary direction — not the modal robustness direction.
    Returns a Series aligned to allp.index.
    """
    if rob_raw is None or rob_raw.empty:
        return pd.Series("—", index=allp.index)

    result = []
    for _, row in allp.iterrows():
        fam_key = _collapse_family(row["family"])
        configs = rob_raw[
            (rob_raw["family"].map(_collapse_family) == fam_key) &
            (rob_raw["task"]   == row["task"]) &
            (rob_raw["metric"] == row["metric"])
        ]
        if configs.empty:
            result.append("—")
            continue
        n_total = len(configs)
        n_match = (configs["rank_order"] == row["rank_order"]).sum()
        result.append(f"{n_match}/{n_total}")
    return pd.Series(result, index=allp.index)


def export_tiered_results(allp: pd.DataFrame,
                          rob_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assign each primary result to an evidence tier (full-robust /
    language-only / unadjusted / effect-only) subject to the effect-size floor,
    attach the robustness order-consistency string, and write the tiered
    group-difference tables (overall and per task) with reader-friendly column
    names. Returns the exported, renamed frame."""
    df = allp.copy()
    df["max_abs_rrb"] = _maxabs_rrb(df)

    # ── Tier assignment ───────────────────────────────────────────────────────
    def _tier(r):
        eff = (_sf(r.get("epsilon2"), 0) >= EPS2_THRESH or
               _sf(r.get("max_abs_rrb"), 0) >= RRB_THRESH)
        if not eff:
            return "none"
        p_full = _sf(r.get("p_mood_adj"),      np.nan)
        p_lang = _sf(r.get("p_mood_lang_adj"), np.nan)
        p_raw  = _sf(r.get("p_kw"),            np.nan)
        if pd.notna(p_full) and p_full < P_THRESH: return "1_full_robust"
        if pd.notna(p_lang) and p_lang < P_THRESH: return "2_language_only"
        if pd.notna(p_raw)  and p_raw  < P_THRESH: return "3_unadjusted"
        return "4_effect_only"

    df["tier"] = df.apply(_tier, axis=1)

    # ── Consistency: how many robustness configs match primary's rank order ───
    df["order_consistency"] = _compute_order_consistency(df, rob_raw)

    # ── Column selection (ordered for readability) ────────────────────────────
    keep = [c for c in [
        "tier", "task", "family", "metric", "rank_order",
        "med_mania", "med_euthymia", "med_depression",
        "epsilon2", "max_abs_rrb",
        "r_rb_ma_eu", "r_rb_ma_de", "r_rb_eu_de",
        "p_kw",
        "p_mood_lang_adj", "p_mood_lang_adj_fdr",
        "p_mood_adj",      "p_mood_adj_fdr",
        "r_wc",
        "order_consistency",
    ] if c in df.columns]

    tiered = df[df["tier"] != "none"].sort_values(
        ["task", "tier", "family", "metric"])[keep]

    # ── Export with reader-friendly names ─────────────────────────────────────
    out = tiered.copy()
    out["tier"] = out["tier"].map(TIER_LABELS).fillna(out["tier"])
    out = out.rename(columns=RENAME_GROUP)
    out.to_csv(OUT / "tiered_group_differences.csv", index=False)
    for task, g in out.groupby("task"):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(task))
        g.to_csv(OUT / f"tiered_{safe}.csv", index=False)

    n1 = (tiered["tier"] == "1_full_robust").sum()
    n2 = (tiered["tier"] == "2_language_only").sum()
    n3 = (tiered["tier"] == "3_unadjusted").sum()
    n4 = (tiered["tier"] == "4_effect_only").sum()
    print(f"\nTiered → {OUT}/tiered_group_differences.csv  "
          f"(T1={n1}, T2={n2}, T3={n3}, T4={n4})")
    for _, r in tiered[tiered["tier"].isin(["1_full_robust","2_language_only"])].iterrows():
        print(f"  [{r['tier']}] {r['task']:<20} {r['family']:<24} {r['metric']:<26} "
              f"ε²={_sf(r.get('epsilon2')):.3f}  "
              f"p_raw={_sf(r.get('p_kw')):.3f}  "
              f"p_lang={_sf(r.get('p_mood_lang_adj')):.3f}  "
              f"p_full={_sf(r.get('p_mood_adj')):.3f}  "
              f"r_wc={_sf(r.get('r_wc')):.2f}  "
              f"{r.get('rank_order','')}  "
              f"consist={r.get('order_consistency','—')}")
    return out


def export_dimensional_results(allp: pd.DataFrame) -> pd.DataFrame:
    """
    YMRS / HDRS severity correlations.
    Entry floor: |r_raw| ≥ 0.30 on the unadjusted correlation AND raw p < .05.
    Three adjustment levels displayed; FDR shown beside adjusted p-values.
    Per-scale tier mirrors the group-difference logic.
    """
    df = allp.copy()

    def _scale_tier(r, scale):
        r_raw = _sf(r.get(f"r_{scale}"), np.nan)
        if pd.isna(r_raw) or abs(r_raw) < RRB_THRESH: return "none"
        p_full = _sf(r.get(f"p_{scale}_full_adj"), np.nan)
        p_lang = _sf(r.get(f"p_{scale}_lang_adj"), np.nan)
        p_raw  = _sf(r.get(f"p_{scale}"),          np.nan)
        if pd.notna(p_full) and p_full < P_THRESH: return "1_full_robust"
        if pd.notna(p_lang) and p_lang < P_THRESH: return "2_language_only"
        if pd.notna(p_raw)  and p_raw  < P_THRESH: return "3_unadjusted"
        return "none"   # effect floor but no p < .05 → not a dimensional finding

    df["ymrs_tier"] = df.apply(lambda r: _scale_tier(r, "ymrs"), axis=1)
    df["hdrs_tier"] = df.apply(lambda r: _scale_tier(r, "hdrs"), axis=1)

    ranked = {"1_full_robust", "2_language_only", "3_unadjusted"}
    dim = df[df["ymrs_tier"].isin(ranked) | df["hdrs_tier"].isin(ranked)].copy()
    if dim.empty:
        print("\nDimensional → none reached |r|≥0.30 with p<.05.")
        return dim

    keep = [c for c in [
        "task", "family", "metric", "rank_order",
        "ymrs_tier",
        "r_ymrs",          "p_ymrs",
        "r_ymrs_lang_adj", "p_ymrs_lang_adj", "p_ymrs_lang_adj_fdr",
        "r_ymrs_full_adj", "p_ymrs_full_adj", "p_ymrs_full_adj_fdr",
        "hdrs_tier",
        "r_hdrs",          "p_hdrs",
        "r_hdrs_lang_adj", "p_hdrs_lang_adj", "p_hdrs_lang_adj_fdr",
        "r_hdrs_full_adj", "p_hdrs_full_adj", "p_hdrs_full_adj_fdr",
        "r_wc", "epsilon2",
    ] if c in dim.columns]
    dim = dim.sort_values(["task", "family", "metric"])[keep]

    out = dim.copy()
    for col in ("ymrs_tier", "hdrs_tier"):
        if col in out.columns:
            out[col] = out[col].map(TIER_LABELS).fillna(out[col])
    out = out.rename(columns={**RENAME_DIM,
                               "ymrs_tier": "mania_severity_tier",
                               "hdrs_tier": "depression_severity_tier"})
    out.to_csv(OUT / "dimensional_severity.csv", index=False)

    n_y = dim["ymrs_tier"].isin(ranked).sum()
    n_h = dim["hdrs_tier"].isin(ranked).sum()
    print(f"\nDimensional → {OUT}/dimensional_severity.csv  "
          f"(YMRS={n_y}, HDRS={n_h})")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_between_patient_stats():
    """Run the full curated between-patient statistical pipeline. Loads each
    feature family, freezes its usable curated features, runs the primary and
    robustness analyses per (family × task), and writes the primary stats,
    tiered group differences, dimensional severity, word-count baseline,
    robustness raw table, feature reference, and JSON summary into OUT."""
    print("═" * 70)
    print("CURATED BETWEEN-PATIENT STATISTICS")
    print(f"tasks={TASKS}  |  FDR within (family × task)")
    print("═" * 70)

    primary_results:    list[pd.DataFrame] = []
    robustness_raw:     list[pd.DataFrame] = []   # per-config long frames
    feature_reference:  list[dict]         = []

    # ── Standard single-CSV families ─────────────────────────────────────────
    for fam, cfg in FAMILY_SOURCES.items():
        path = cfg["path"]
        if not path.exists():
            print(f"\n[{fam}] missing: {path} — skip"); continue
        print(f"\n[{fam}]")
        raw  = _norm_state(pd.read_csv(path))
        keep = KEEP_LISTS[cfg["keep"]]
        fam_tasks = _tasks_for(fam)

        primary_by_task: dict[str, pd.DataFrame] = {}
        for task in fam_tasks:
            dprim = (_apply_primary(raw, fam, task) if cfg["primary"]
                     else raw[raw["task"].astype(str) == task] if "task" in raw.columns
                     else raw)
            if not dprim.empty:
                primary_by_task[task] = dprim

        frozen, per_task_excl = _freeze_features(primary_by_task, keep, fam)
        if not frozen:
            print(f"  [{fam}] no features — skip"); continue

        for task, dprim in primary_by_task.items():
            res = _run_primary(dprim, frozen, fam, task)
            if res is not None:
                primary_results.append(res)

        rob_col = {"speech_graphs": "experiment", "semantic_networks": "threshold"}.get(fam)
        if rob_col:
            keep_frozen = {f: keep[f] for f in frozen if f in keep}
            for task in fam_tasks:
                rob = _run_robustness(raw, keep_frozen, fam, task, rob_col)
                if rob is not None:
                    robustness_raw.append(rob)

        feature_reference.extend(
            _build_feature_rows(fam, keep, frozen, per_task_excl, fam_tasks)
        )


    # ── Co-occurrence ─────────────────────────────────────────────────────────
    if COOC_PATH.exists():
        print(f"\n[cooccurrence]")
        cooc = _norm_state(pd.read_csv(COOC_PATH))
        for sub_fam in ("cooccurrence_node", "cooccurrence_triad"):
            keep = KEEP_LISTS[sub_fam]
            primary_by_task = {task: _apply_primary(cooc, "cooccurrence_emotion", task)
                               for task in TASKS}
            primary_by_task = {t: d for t, d in primary_by_task.items() if not d.empty}
            frozen, per_task_excl = _freeze_features(primary_by_task, keep, sub_fam)

            if not frozen: continue
            for task, dprim in primary_by_task.items():
                res = _run_primary(dprim, frozen, sub_fam, task)
                if res is not None: primary_results.append(res)
            keep_frozen = {f: keep[f] for f in frozen if f in keep}
            for task in TASKS:
                rob = _run_robustness(cooc, keep_frozen, sub_fam, task, "window")
                if rob is not None: robustness_raw.append(rob)
            feature_reference.extend(
                _build_feature_rows(sub_fam, keep, frozen, per_task_excl, TASKS)
            )
    else:
        print(f"\n[cooccurrence] missing: {COOC_PATH} — skip")

    # ── Topic ─────────────────────────────────────────────────────────────────
    keep = KEEP_LISTS["topic"]
    topic_frames: dict[str, pd.DataFrame] = {}
    for model in TOPIC_MODELS:
        p = _RES / "topic_analysis" / model / "topics_shape_patient_values.csv"
        if p.exists():
            tm = _norm_state(pd.read_csv(p))
            if "model" not in tm.columns: tm["model"] = model
            topic_frames[model] = tm
        else:
            print(f"\n[topic:{model}] missing — skip")

    if topic_frames:
        print(f"\n[topic] models: {list(topic_frames)}")
        pooled = {f"{m}_{t}": tm[tm[next((c for c in ("task","task_set") if c in tm.columns),"task")].astype(str)==t]
                  for m, tm in topic_frames.items() for t in EMBEDDING_TASKS
                  if not tm[tm[next((c for c in ("task","task_set") if c in tm.columns),"task")].astype(str)==t].empty}
        frozen, per_task_excl = _freeze_features(pooled, keep, "topic")
        if frozen:
            per_model_long: list[pd.DataFrame] = []
            for model, tm in topic_frames.items():
                tcol = next((c for c in ("task","task_set") if c in tm.columns), "task")
                for task in EMBEDDING_TASKS:
                    dprim = tm[tm[tcol].astype(str) == task]
                    if dprim.empty: continue
                    res = _run_primary(dprim, frozen, f"topic_{model}", task)
                    if res is not None:
                        primary_results.append(res)
                        rr = res.copy(); rr["config"] = model
                        per_model_long.append(rr)
            # topic robustness = per-model long frame (models are the configs)
            if per_model_long:
                long = pd.concat(per_model_long, ignore_index=True)
                long["family"] = "topic"
                robustness_raw.append(long)
            for model in topic_frames:
                model_excl = {}
                for task in EMBEDDING_TASKS:
                    tm = topic_frames[model]
                    tcol = next((c for c in ("task","task_set") if c in tm.columns), "task")
                    subset = tm[tm[tcol].astype(str) == task]
                    if not subset.empty:
                        _, excl = _feature_cols(_prep(subset), keep, f"[topic_{model}|{task}] ")
                        model_excl[task] = excl
                    else:
                        model_excl[task] = {f: "task-not-run" for f in keep}
                feature_reference.extend(
                    _build_feature_rows(f"topic_{model}", keep, frozen, model_excl, EMBEDDING_TASKS)
                )
    else:
        print("\n[topic] no model CSVs — skip")

    # ── Semantic similarity ───────────────────────────────────────────────────
    if SEMSIM_PATH.exists():
        print(f"\n[semantic_similarity]")
        ss   = _norm_state(pd.read_csv(SEMSIM_PATH))
        keep = KEEP_LISTS["semantic_similarity"]

        # Word-level (XLM-R primary, fastText robustness)
        word_by_task = {task: _apply_primary(ss, "semantic_similarity", task)
                        for task in EMBEDDING_TASKS}
        word_by_task = {t: d for t, d in word_by_task.items() if not d.empty}
        if word_by_task:
            frozen_w, per_task_excl_w   = _freeze_features(word_by_task, keep, "semantic_similarity_word")
            if frozen_w:
                for task, dprim in word_by_task.items():
                    res = _run_primary(dprim, frozen_w, "semantic_similarity_word", task)
                    if res is not None: primary_results.append(res)
                if "source" in ss.columns:
                    wl = ss[ss["source"].astype(str).str.contains("xlm|fasttext", case=False)]
                    for task in EMBEDDING_TASKS:
                        rob = _run_robustness(wl, {f: keep[f] for f in frozen_w if f in keep},
                                              "semantic_similarity_word", task, "source")
                        if rob is not None: robustness_raw.append(rob)
            feature_reference.extend(
                _build_feature_rows("semantic_similarity_word", keep, frozen_w, per_task_excl_w, EMBEDDING_TASKS)
            )

        # Sentence-level (per-model, models are the configs)
        if "source" in ss.columns:
            sent_frames = {m: ss[ss["source"].astype(str).str.contains(m, case=False)]
                           for m in SENTENCE_MODELS}
            sent_frames = {m: d for m, d in sent_frames.items() if not d.empty}
            if sent_frames:
                if per_model_long:
                    long = pd.concat(per_model_long, ignore_index=True)
                    long["family"] = "semantic_similarity_sentence"
                    robustness_raw.append(long)
                for m, sm in sent_frames.items():
                    model_by_task = {}
                    tcol = next((c for c in ("task","task_set") if c in sm.columns), "task")
                    for task in EMBEDDING_TASKS:
                        s = sm[sm[tcol].astype(str) == task]
                        if not s.empty:
                            model_by_task[task] = s

                    frozen_m, per_task_excl_m = _freeze_features(model_by_task, keep,
                                                                f"semantic_similarity_sent_{m}")
                    if frozen_m:
                        for task in EMBEDDING_TASKS:
                            dprim = model_by_task.get(task)
                            if dprim is None or dprim.empty:
                                continue
                            res = _run_primary(dprim, frozen_m,
                                            f"semantic_similarity_sent_{m}", task)
                            if res is not None:
                                primary_results.append(res)
                                rr = res.copy(); rr["config"] = m
                                per_model_long.append(rr)
                    
                    feature_reference.extend(
                        _build_feature_rows(f"semantic_similarity_sent_{m}", keep,
                                            frozen_m, per_task_excl_m, EMBEDDING_TASKS)
                    )
                    
    else:
        print(f"\n[semantic_similarity] missing — skip")

    # ── Word count verbosity baseline ─────────────────────────────────────────
    lex_path = FAMILY_SOURCES["lexical"]["path"]
    if lex_path.exists():
        lex_raw = _norm_state(pd.read_csv(lex_path))
        wc_by_task = {t: lex_raw[lex_raw["task"].astype(str) == t]
                      for t in TASKS if (lex_raw["task"].astype(str) == t).any()}
        wc = _wordcount_analysis(wc_by_task)
        if not wc.empty:
            wc.to_csv(OUT / "word_count_verbosity.csv", index=False)
            print(f"\nWord count → {OUT}/word_count_verbosity.csv")
            print(wc[["task","rank_order","epsilon2","p_kw"]].to_string(index=False))

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)

    allr_raw = pd.concat(robustness_raw, ignore_index=True) if robustness_raw else None

    if primary_results:
        allp = strip_sig_flags(pd.concat(primary_results, ignore_index=True))
        for (fam, task), g in allp.groupby(["family", "task"]):
            g.to_csv(OUT / f"{fam}_{task}_stats.csv", index=False)
        allp.to_csv(OUT / "ALL_primary_stats.csv", index=False)
        print(f"Primary stats  → {OUT}/ALL_primary_stats.csv")
        export_tiered_results(allp, allr_raw)
        export_dimensional_results(allp)
    else:
        print("⚠ No primary results — check paths/schemas.")

    if allr_raw is not None:
        allr_raw.to_csv(OUT / "ALL_robustness_raw.csv", index=False)
        print(f"Robustness raw → {OUT}/ALL_robustness_raw.csv")

    pd.DataFrame(feature_reference).to_csv(OUT / "curated_feature_reference.csv", index=False)
    print(f"Feature ref    → {OUT}/curated_feature_reference.csv")

    summary = {
        "tasks": TASKS, "embedding_tasks": EMBEDDING_TASKS,
        "fdr_scope": "within (family × task), primary config only",
        "effect_floor": f"eps2>={EPS2_THRESH} OR max|r_rb|>={RRB_THRESH}",
        "tier_logic": "T1 lang+wc adj, T2 lang adj, T3 unadjusted, T4 effect-only; FDR displayed not gated",
        "order_consistency": "X/N configs matching primary experiment's exact rank order",
        "primary_configs": PRIMARY,
        "topic_models": TOPIC_MODELS,
        "sentence_models": SENTENCE_MODELS,
    }
    with (OUT / "between_patient_curated_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary        → {OUT}/between_patient_curated_summary.json")
    print("═" * 70)


if __name__ == "__main__":
    run_between_patient_stats()