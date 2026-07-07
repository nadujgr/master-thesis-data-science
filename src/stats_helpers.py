"""
Mood-state analysis helpers for speech graph metrics. For each metric ×
experiment, produces one row combining: Kruskal-Wallis omnibus and pairwise
rank-biserial effect sizes; confound diagnostics (word-count correlation,
within-euthymia language test) and a rank-based ANCOVA adjusting for language
and verbosity; severity correlations; and per-state medians with rank order.
A second helper summarises consistency across experiments (significance counts
and rank-order stability). Returns two DataFrames per analysis level.
"""

from __future__ import annotations

from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

STATES = ["mania", "euthymia", "depression"]
PAIRS = [("mania", "euthymia"), ("mania", "depression"), ("euthymia", "depression")]


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def fold_mixed(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed states → mania, with a notice."""
    n = int((df["symptom_state"] == "mixed").sum())
    if n:
        df = df.copy()
        df["symptom_state"] = df["symptom_state"].replace("mixed", "mania")
        print(f"  Folded {n} 'mixed' → 'mania'")
    return df

def classify_pattern(rank_order: str) -> str:
    """Classify by which pole is elevated."""
    if pd.isna(rank_order) or rank_order is None:
        return None
    
    states = rank_order.split(">")
    if len(states) != 3:
        return None
    
    highest = states[0]
    lowest = states[-1]

    if highest == "mania":
        return "mania_elevated"
    elif highest == "depression":
        return "depression_elevated"
    elif highest == "euthymia":
        if lowest == "mania":
            return "mania_reduced"
        elif lowest == "depression":
            return "depression_reduced"
        else:
            return "euthymia_elevated"
    return "mixed"



def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(1 - 2 * u / (len(a) * len(b)))


def _fdr(ps: pd.Series) -> pd.Series:
    """BH-FDR correction, NaN-safe. Returns same-shape Series."""
    out = pd.Series(np.nan, index=ps.index, dtype=float)
    valid = ps.notna()
    if valid.sum() == 0:
        return out
    _, q, _, _ = multipletests(ps[valid].values, method="fdr_bh")
    out.loc[valid] = q
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment analysis: one row per metric, all stats inline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_experiment(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    One DataFrame with one row per metric. All per-experiment stats inline.
    Expects columns: symptom_state, language, word_count, + the metrics.
    """
    states_present = [s for s in STATES if s in df["symptom_state"].values]
    groups = {s: df[df["symptom_state"] == s] for s in states_present}

    has_lang = "language" in df.columns and df["language"].nunique() > 1
    has_wc   = "word_count" in df.columns and df["word_count"].notna().any()

    rows = []
    for m in metrics:
        if m not in df.columns or df[m].nunique() < 2:
            continue

        row = {"metric": m}

        # ── group descriptives ─────────────────────────────────────────────
        meds = {}
        for s in STATES:
            if s in groups:
                vals = groups[s][m].dropna()
                row[f"med_{s}"] = round(vals.median(), 4) if len(vals) else np.nan
                row[f"n_{s}"]   = int(len(vals))
                if len(vals):
                    meds[s] = vals.median()
            else:
                row[f"med_{s}"] = np.nan
                row[f"n_{s}"]   = 0
        row["rank_order"] = (">".join(sorted(meds, key=meds.get, reverse=True))
                             if len(meds) == len(states_present) else None)

        # ── omnibus KW ────────────────────────────────────────────────────
        arrays = [groups[s][m].dropna().values for s in states_present]
        arrays = [a for a in arrays if len(a) >= 3]
        if len(arrays) >= 2:
            try:
                H, p_kw = kruskal(*arrays)
                k, n_total = len(arrays), sum(len(a) for a in arrays)
                row["p_kw"]     = float(p_kw)
                row["epsilon2"] = round(max(0.0, (H - k + 1) / (n_total - k)), 4)
            except Exception:
                row["p_kw"], row["epsilon2"] = np.nan, np.nan
        else:
            row["p_kw"], row["epsilon2"] = np.nan, np.nan

        # ── pairwise effect sizes + raw MW p-values ───────────────────────
        for s1, s2 in PAIRS:
            label = f"{s1[:2]}_{s2[:2]}"
            if s1 in groups and s2 in groups:
                a = groups[s1][m].dropna().values
                b = groups[s2][m].dropna().values
                if len(a) >= 3 and len(b) >= 3:
                    row[f"r_rb_{label}"] = round(_rank_biserial(a, b), 4)
                    row[f"p_mw_{label}"] = round(
                        float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)
                else:
                    row[f"r_rb_{label}"] = np.nan
                    row[f"p_mw_{label}"] = np.nan
            else:
                row[f"r_rb_{label}"] = np.nan
                row[f"p_mw_{label}"] = np.nan

        # ── symptom severity correlations ─────────────────────────────────────
        # Three Spearman versions per scale, FDR'd separately:
        #   raw      : no adjustment
        #   lang_adj : partial, language residualised out
        #   full_adj : partial, language + word count residualised
        for scale, scale_col in [("ymrs", "ymrs"), ("hdrs", "hdrs")]:
            if scale_col in df.columns and df[scale_col].notna().sum() >= 10:
                sev_cols = ([m, scale_col]
                            + (["language"]   if has_lang else [])
                            + (["word_count"] if has_wc   else []))
                pair = df[sev_cols].dropna()
                if len(pair) >= 10:
                    # (1) Raw
                    rho_raw, p_raw = spearmanr(pair[m], pair[scale_col])
                    row[f"r_{scale}"]          = round(float(rho_raw), 4)
                    row[f"p_{scale}"]          = round(float(p_raw),   4)

                    # (2) Language-partialled
                    if has_lang and pair["language"].nunique() > 1:
                        try:
                            resid_m_lang = smf.ols(
                                f"Q('{m}') ~ C(language)", data=pair).fit().resid
                            resid_s_lang = smf.ols(
                                f"Q('{scale_col}') ~ C(language)", data=pair).fit().resid
                            rho_lang, p_lang = spearmanr(resid_m_lang, resid_s_lang)
                        except Exception:
                            rho_lang, p_lang = rho_raw, p_raw
                    else:
                        rho_lang, p_lang = rho_raw, p_raw
                    row[f"r_{scale}_lang_adj"] = round(float(rho_lang), 4)
                    row[f"p_{scale}_lang_adj"] = round(float(p_lang),   4)

                    # (3) Language + word count partialled (full adjustment)
                    cov_terms_sev = []
                    if has_lang and pair["language"].nunique() > 1:
                        cov_terms_sev.append("C(language)")
                    if has_wc and "word_count" in pair.columns:
                        cov_terms_sev.append("word_count")
                    if cov_terms_sev:
                        try:
                            cov_str_sev = " + ".join(cov_terms_sev)
                            resid_m_full = smf.ols(
                                f"Q('{m}') ~ {cov_str_sev}", data=pair).fit().resid
                            resid_s_full = smf.ols(
                                f"Q('{scale_col}') ~ {cov_str_sev}", data=pair).fit().resid
                            rho_full, p_full = spearmanr(resid_m_full, resid_s_full)
                        except Exception:
                            rho_full, p_full = rho_lang, p_lang
                    else:
                        rho_full, p_full = rho_lang, p_lang
                    row[f"r_{scale}_full_adj"] = round(float(rho_full), 4)
                    row[f"p_{scale}_full_adj"] = round(float(p_full),   4)

                else:
                    for suffix in ["", "_lang_adj", "_full_adj"]:
                        row[f"r_{scale}{suffix}"] = np.nan
                        row[f"p_{scale}{suffix}"] = np.nan
            else:
                for suffix in ["", "_lang_adj", "_full_adj"]:
                    row[f"r_{scale}{suffix}"] = np.nan
                    row[f"p_{scale}{suffix}"] = np.nan

        # ── confound diagnostics ──────────────────────────────────────────
        # Verbosity: Spearman with word count
        if has_wc:
            pair = df[[m, "word_count"]].dropna()
            if len(pair) >= 5 and m != "word_count":
                corr, _ = spearmanr(pair[m], pair["word_count"])
                row["r_wc"] = round(float(corr), 4)
            elif m == "word_count":
                row["r_wc"] = 1.0
            else:
                row["r_wc"] = np.nan
        else:
            row["r_wc"] = np.nan

        # Language: KW within euthymics only (cleanest baseline)
        if has_lang and "euthymia" in groups:
            eu = groups["euthymia"]
            lang_arrs = [eu[eu["language"] == lg][m].dropna().values
                         for lg in eu["language"].unique()]
            lang_arrs = [a for a in lang_arrs if len(a) >= 3]
            if len(lang_arrs) >= 2:
                try:
                    row["p_lang_eu"] = round(float(kruskal(*lang_arrs).pvalue), 4)
                except Exception:
                    row["p_lang_eu"] = np.nan
            else:
                row["p_lang_eu"] = np.nan
        else:
            row["p_lang_eu"] = np.nan

        # ── confound-adjusted mood test (rank-based ANCOVA) ───────────────
        # Rank-transforming y makes OLS robust to non-normality / outliers
        # (relevant at mania n ≈ 10). The mood effect is F-tested against a
        # reduced model without mood; language is F-tested similarly.
        cov_cols = [c for c in (["language"] if has_lang else []) +
                                (["word_count"] if has_wc else [])]
        sub = df[["symptom_state", m] + cov_cols].dropna()
        sub = sub[sub["symptom_state"].isin(states_present)]
        row["p_mood_adj"]      = np.nan
        row["p_mood_lang_adj"] = np.nan
        row["p_lang_adj"]      = np.nan
        row["r2_full"]         = np.nan
        if len(sub) >= 10 and sub["symptom_state"].nunique() >= 2:
            try:
                sub = sub.copy()
                sub["_y"] = sub[m].rank()
                cov_terms = []
                if "language"   in cov_cols: cov_terms.append("C(language)")
                if "word_count" in cov_cols: cov_terms.append("word_count")
                cov_str = " + " + " + ".join(cov_terms) if cov_terms else ""

                full = smf.ols(f"_y ~ C(symptom_state){cov_str}", data=sub).fit()
                row["r2_full"] = round(float(full.rsquared), 4)

                # Test mood: full model vs no-mood (language + word_count)
                reduced_mood = smf.ols(
                    f"_y ~ {' + '.join(cov_terms)}" if cov_terms else "_y ~ 1",
                    data=sub).fit()
                _, p_mood, _ = full.compare_f_test(reduced_mood)
                row["p_mood_adj"] = float(p_mood)

                # Mood test with language only (no word_count): is mood
                # significant after language adjustment alone? Relevant for
                # families with internal verbosity control (windowed speech
                # graphs, z-scored metrics).
                if has_lang:
                    lang_only_full = smf.ols(
                        f"_y ~ C(symptom_state) + C(language)", data=sub).fit()
                    lang_only_reduced = smf.ols(
                        f"_y ~ C(language)", data=sub).fit()
                    _, p_mood_lang, _ = lang_only_full.compare_f_test(
                        lang_only_reduced)
                    row["p_mood_lang_adj"] = float(p_mood_lang)

                # Test language: mood+language vs mood only
                if has_lang:
                    try:
                        lang_full    = smf.ols(
                            f"_y ~ C(symptom_state) + C(language)", data=sub).fit()
                        lang_reduced = smf.ols(
                            f"_y ~ C(symptom_state)", data=sub).fit()
                        _, p_lang, _ = lang_full.compare_f_test(lang_reduced)
                        row["p_lang_adj"] = float(p_lang)
                    except Exception:
                        row["p_lang_adj"] = np.nan
            except Exception:
                pass

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # FDR within experiment, on each p-value family separately
    out["p_ymrs_fdr"]          = _fdr(out["p_ymrs"]).round(4)
    out["p_hdrs_fdr"]          = _fdr(out["p_hdrs"]).round(4)
    out["p_ymrs_lang_adj_fdr"] = _fdr(out["p_ymrs_lang_adj"]).round(4)
    out["p_hdrs_lang_adj_fdr"] = _fdr(out["p_hdrs_lang_adj"]).round(4)
    out["p_ymrs_full_adj_fdr"] = _fdr(out["p_ymrs_full_adj"]).round(4)
    out["p_hdrs_full_adj_fdr"] = _fdr(out["p_hdrs_full_adj"]).round(4)
    out["p_kw_fdr"]       = _fdr(out["p_kw"]).round(4)
    out["p_mood_adj_fdr"] = _fdr(out["p_mood_adj"]).round(4)
    out["p_mood_lang_adj_fdr"] = _fdr(out["p_mood_lang_adj"]).round(4)

    out["sig_unadj"] = out["p_kw_fdr"]            < 0.05
    out["sig_adj"]   = out["p_mood_adj_fdr"]      < 0.05
    out["sig_ymrs"]  = out["p_ymrs_fdr"]          < 0.05
    out["sig_hdrs"]  = out["p_hdrs_fdr"]          < 0.05

    out["p_kw"]            = out["p_kw"].round(4)
    out["p_mood_adj"]      = out["p_mood_adj"].round(4)
    out["p_mood_lang_adj"] = out["p_mood_lang_adj"].round(4)
    out["p_lang_adj"]      = out["p_lang_adj"].round(4)
    out["p_ymrs"]          = out["p_ymrs"].round(4)
    out["p_hdrs"]          = out["p_hdrs"].round(4)
    out["p_ymrs_lang_adj"] = out["p_ymrs_lang_adj"].round(4)
    out["p_hdrs_lang_adj"] = out["p_hdrs_lang_adj"].round(4)
    out["p_ymrs_full_adj"] = out["p_ymrs_full_adj"].round(4)
    out["p_hdrs_full_adj"] = out["p_hdrs_full_adj"].round(4)
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-experiment consistency
# ─────────────────────────────────────────────────────────────────────────────

def consistency_across_experiments(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    long_df: per-experiment results stacked, with an 'experiment' column.
    Returns one row per metric with BOTH robustness criteria.
    """
    rows = []
    for metric, grp in long_df.groupby("metric"):
        n_exp = len(grp)
        n_sig_unadj = int(grp["sig_unadj"].fillna(False).sum())
        n_sig_adj   = int(grp["sig_adj"].fillna(False).sum())
        
        # ── exact rank order consistency ──────────────────────────────────
        orders = grp["rank_order"].dropna()
        if len(orders) > 0:
            counts = orders.value_counts()
            most_common_order  = counts.index[0]
            n_consistent_order = int(counts.iloc[0])
            pct_consistent_order = n_consistent_order / len(orders)
        else:
            most_common_order, n_consistent_order, pct_consistent_order = None, 0, np.nan
        
        # ── pole-based consistency ────────────────────────────────────────
        patterns = grp["rank_order"].apply(classify_pattern).dropna()
        if len(patterns) > 0:
            pattern_counts = patterns.value_counts()
            most_common_pattern = pattern_counts.index[0]
            n_consistent_pattern = int(pattern_counts.iloc[0])
            pct_consistent_pattern = n_consistent_pattern / len(patterns)
        else:
            most_common_pattern, n_consistent_pattern, pct_consistent_pattern = None, 0, np.nan
        
        # ── Effect sizes ──────────────────────────────────────────────────
        pair_means = {}
        for pair_label in ("ma_eu", "ma_de", "eu_de"):
            col = f"r_rb_{pair_label}"
            pair_means[pair_label] = (grp[col].abs().mean()
                                      if col in grp.columns and grp[col].notna().any()
                                      else np.nan)
        
        mean_eps2 = (grp["epsilon2"].mean()
                     if "epsilon2" in grp.columns and grp["epsilon2"].notna().any()
                     else np.nan)
        
        # ── Confound diagnostics ──────────────────────────────────────────
        mean_abs_r_wc = (grp["r_wc"].abs().mean()
                         if "r_wc" in grp.columns else np.nan)
        
        n_lang_eu_sig  = (int((grp["p_lang_eu"]  < 0.05).fillna(False).sum())
                          if "p_lang_eu"  in grp.columns else 0)
        n_lang_adj_sig = (int((grp["p_lang_adj"] < 0.05).fillna(False).sum())
                          if "p_lang_adj" in grp.columns else 0)
        
        # ── Severity correlations ─────────────────────────────────────────
        n_ymrs_sig = (int((grp["p_ymrs_fdr"] < 0.05).fillna(False).sum())
                      if "p_ymrs_fdr" in grp.columns else 0)
        n_hdrs_sig = (int((grp["p_hdrs_fdr"] < 0.05).fillna(False).sum())
                      if "p_hdrs_fdr" in grp.columns else 0)
        
        mean_r_ymrs = (round(grp["r_ymrs"].abs().mean(), 4)
                       if "r_ymrs" in grp.columns and grp["r_ymrs"].notna().any()
                       else np.nan)
        mean_r_hdrs = (round(grp["r_hdrs"].abs().mean(), 4)
                       if "r_hdrs" in grp.columns and grp["r_hdrs"].notna().any()
                       else np.nan)
        
        # ── TWO robustness criteria ───────────────────────────────────────
        # Strict: exact rank order
        robust_strict = (
            n_sig_adj >= max(2, n_exp // 2)
            and (pct_consistent_order >= 0.70 if not pd.isna(pct_consistent_order) else False)
        )
        
        # Relaxed: pole-based pattern
        robust_pole = (
            n_sig_adj >= max(2, n_exp // 2)
            and (pct_consistent_pattern >= 0.60 if not pd.isna(pct_consistent_pattern) else False)
        )
        
        rows.append({
            "metric":                   metric,
            "n_experiments":            n_exp,
            "n_sig_unadj_fdr":          n_sig_unadj,
            "n_sig_adj_fdr":            n_sig_adj,
            # Exact order
            "most_common_order":        most_common_order,
            "n_consistent_order":       n_consistent_order,
            "pct_consistent_order":     round(pct_consistent_order, 3) if not pd.isna(pct_consistent_order) else np.nan,
            # Pole-based pattern
            "most_common_pattern":      most_common_pattern,
            "n_consistent_pattern":     n_consistent_pattern,
            "pct_consistent_pattern":   round(pct_consistent_pattern, 3) if not pd.isna(pct_consistent_pattern) else np.nan,
            # Effect sizes
            "mean_epsilon2":            round(mean_eps2, 4) if not pd.isna(mean_eps2) else np.nan,
            "mean_abs_r_rb_ma_eu":      round(pair_means["ma_eu"], 4) if not pd.isna(pair_means["ma_eu"]) else np.nan,
            "mean_abs_r_rb_ma_de":      round(pair_means["ma_de"], 4) if not pd.isna(pair_means["ma_de"]) else np.nan,
            "mean_abs_r_rb_eu_de":      round(pair_means["eu_de"], 4) if not pd.isna(pair_means["eu_de"]) else np.nan,
            # Confounds
            "mean_abs_r_wc":            round(mean_abs_r_wc, 4) if not pd.isna(mean_abs_r_wc) else np.nan,
            "n_exp_lang_eu":            n_lang_eu_sig,
            "n_exp_lang_adj":           n_lang_adj_sig,
            # Severity gradients
            "n_exp_ymrs_sig":           n_ymrs_sig,
            "n_exp_hdrs_sig":           n_hdrs_sig,
            "mean_abs_r_ymrs":          mean_r_ymrs,
            "mean_abs_r_hdrs":          mean_r_hdrs,
            # Robustness flags
            "robust_strict":            robust_strict,
            "robust_pole":              robust_pole,
        })
    
    return (pd.DataFrame(rows)
            .sort_values(["robust_pole", "n_sig_adj_fdr", "mean_epsilon2"],
                         ascending=[False, False, False])
            .reset_index(drop=True))





# ─────────────────────────────────────────────────────────────────────────────
# Pretty printers
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_summary(name: str, df_exp: pd.DataFrame) -> None:
    n_sig_unadj = int(df_exp["sig_unadj"].sum())
    n_sig_adj   = int(df_exp["sig_adj"].sum())
    print(f"\n  {name}")
    print(f"  KW FDR-sig: {n_sig_unadj}/{len(df_exp)}   "
          f"adjusted FDR-sig: {n_sig_adj}/{len(df_exp)}")
    if n_sig_adj > 0:
        cols = ["metric", "rank_order", "epsilon2",
                "r_rb_ma_de", "p_kw_fdr", "p_mood_adj_fdr", "r_wc"]
        cols = [c for c in cols if c in df_exp.columns]
        print(df_exp[df_exp["sig_adj"]]
              .sort_values("p_mood_adj_fdr")[cols]
              .to_string(index=False))


def print_consistency_summary(summary: pd.DataFrame) -> None:
    n_strict = int(summary['robust_strict'].sum())
    n_pole = int(summary['robust_pole'].sum())
    n_pole_only = int((summary['robust_pole'] & ~summary['robust_strict']).sum())
    
    print(f"\n  ROBUSTNESS SUMMARY:")
    print(f"    Strict (exact order ≥70%):        {n_strict}/{len(summary)}")
    print(f"    Pole-based (pole pattern ≥60%):   {n_pole}/{len(summary)}")
    print(f"    Pole-only (not strict):           {n_pole_only}/{len(summary)}")
    
    cols = ["metric", "n_sig_adj_fdr", "n_experiments",
            "most_common_pattern", "pct_consistent_pattern",
            "most_common_order", "pct_consistent_order",
            "mean_epsilon2",
            "mean_abs_r_rb_ma_de",
            "n_exp_ymrs_sig", "n_exp_hdrs_sig",
            "robust_strict", "robust_pole"]
    cols = [c for c in cols if c in summary.columns]
    print("\n" + summary[cols].to_string(index=False))