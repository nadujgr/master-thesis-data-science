"""
Mood-state classification from Speech Graph Attributes on the emotion task.

Loads the speech-graph emotion patient table (the methodology/task combination
with the strongest word-count- and language-adjusted mood signal) and runs
3-way (mania/euthymia/depression) and binary contrasts with fixed, lightly
regularised classifiers (dummy, NaiveBayes, LogisticRegression, linear SVM)
under leave-one-out CV. Language is residualised inside each fold via one-hot
dummies (no leakage); zero-variance features are dropped. Reports balanced
accuracy, F1, kappa, ROC/PR-AUC with bootstrap CIs, a word-count-only baseline,
and LOO-aligned permutation and SHAP feature importance. Results are written as
CSVs and a JSON summary to OUTPUT_DIR.
"""

import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import RESULTS_DIR
from src.stats_helpers import fold_mixed

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RANDOM_STATE        = 42
N_BOOTSTRAP         = 1000
N_PERM_REPEATS      = 50   # draws per fold for LOO permutation importance
SYMPTOM_STATE_ORDER = ["mania", "euthymia", "depression"]
CHANCE_3WAY         = 1.0 / 3
CHANCE_BINARY       = 0.5

_RES       = Path(RESULTS_DIR)
OUTPUT_DIR = _RES / "classification" / "speech_graphs_emotion"

# ── Data paths ─────────────────────────────────────────────────────
PATIENTS_CSV = _RES / "speech_graphs" / "speech_graphs_patient_values.csv"

# ── Feature specification ──────────────────────────────────────────
SGA_METRICS_ALL = [
    "LSC", "Density",
    "ASP", "CC", "RE", "L1", "L2",
]

# ── Positive-class definitions for binary contrasts ────────────────
CONTRAST_POSITIVE_CLASS = {
    ("mania",    "depression"): "mania",
    ("mania",    "euthymia"):   "mania",
    ("euthymia", "depression"): "euthymia",
}

# Task and experiment identifiers
TASK               = "emotion"
EXPERIMENT_COL     = "task"
PRIMARY_EXPERIMENT = "emotion_w10s1"
PRIMARY_EXP_COL    = "experiment"


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════

CLASSIFIERS = {
    "Dummy_stratified": Pipeline([
        ("clf", DummyClassifier(strategy="stratified",
                                random_state=RANDOM_STATE)),
    ]),
    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000,
            random_state=RANDOM_STATE)),
    ]),
    "SVM_linear": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0,
                    class_weight="balanced", probability=True,
                    random_state=RANDOM_STATE)),
    ]),
}


# ═══════════════════════════════════════════════════════════════════
# LANGUAGE RESIDUALISATION HELPER
# ═══════════════════════════════════════════════════════════════════

def _residualise(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Regress each column of X on covariate matrix C and return residuals.
    C must be 2D (n_samples, n_covariates).
    Columns with fewer than 5 non-NaN values are left unchanged.
    X is not modified in place; a copy is returned.
    """
    X = X.copy()
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    for i in range(X.shape[1]):
        col   = X[:, i]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        lr = LinearRegression().fit(C[valid], col[valid])
        X[valid, i] = col[valid] - lr.predict(C[valid])
    return X


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_speech_graphs_emotion() -> dict | None:
    """Load and prepare the speech-graph emotion-task patient table for
    classification. Restricts to the emotion task and primary window experiment,
    folds mixed episodes, keeps the three analysed mood states, drops rows with
    missing SGA features and any zero-variance feature, and builds language dummy
    columns for per-fold residualisation. Returns a dict bundling the feature
    matrix, labels, feature list, patient frame and language dummies, or None if
    the data cannot be loaded or is unusable."""
    if not PATIENTS_CSV.exists():
        print(f"✗ Patient CSV not found: {PATIENTS_CSV}")
        return None

    patients = pd.read_csv(PATIENTS_CSV)

    if "symptom_state" not in patients.columns and "episode_type" in patients.columns:
        patients = patients.rename(columns={"episode_type": "symptom_state"})
    if "task" not in patients.columns and "task_set" in patients.columns:
        patients = patients.rename(columns={"task_set": "task"})

    if EXPERIMENT_COL in patients.columns:
        patients = patients[patients[EXPERIMENT_COL] == TASK].copy()
    else:
        print(f"  ⚠ Column '{EXPERIMENT_COL}' not found — using all rows.")

    if PRIMARY_EXP_COL in patients.columns:
        before = len(patients)
        patients = patients[patients[PRIMARY_EXP_COL] == PRIMARY_EXPERIMENT].copy()
        print(f"  Restricted to experiment='{PRIMARY_EXPERIMENT}': "
              f"{before} → {len(patients)} rows")
    else:
        print(f"  ⚠ Column '{PRIMARY_EXP_COL}' not found — window not filtered.")

    patients["caliber_code"] = (patients["caliber_code"]
                                .astype(str).str.strip().str.lower())
    patients = fold_mixed(patients)
    patients = patients[patients["symptom_state"].isin(SYMPTOM_STATE_ORDER)]
    patients = patients.drop_duplicates("caliber_code").copy()

    feats = [f for f in SGA_METRICS_ALL if f in patients.columns]
    if not feats:
        print("✗ None of the SGA metrics found in patient CSV.")
        return None

    patients = patients.loc[patients[feats].notna().all(axis=1)].reset_index(drop=True)
    if patients.empty:
        print("✗ No patients remaining after NaN drop.")
        return None

    zero_var = [f for f in feats if patients[f].nunique() <= 1]
    if zero_var:
        print(f"  Zero-variance filter: dropped {zero_var}")
        feats = [f for f in feats if f not in zero_var]
    if len(feats) < 2:
        print("✗ Fewer than 2 features remaining after zero-variance filter.")
        return None

    print(f"\n  Loaded: n={len(patients)} patients  "
          f"({dict(patients['symptom_state'].value_counts())})")
    print(f"  Features ({len(feats)}): {', '.join(feats)}")

    lang_dummies = None
    if "language" in patients.columns and patients["language"].nunique() > 1:
        lang_dummies = pd.get_dummies(
            patients["language"].astype(str), drop_first=True
        ).values.astype(float)
        n_lang = patients["language"].nunique()
        print(f"  Language: {n_lang} unique values → "
              f"{lang_dummies.shape[1]} dummy column(s), "
              f"residualised inside LOO folds.")
    else:
        print("  ⚠ Language missing or single-valued — residualisation skipped.")

    return {
        "caliber_code":      patients["caliber_code"].tolist(),
        "symptom_state":     patients["symptom_state"].values,
        "X":                 patients[feats].copy(),
        "selected_features": feats,
        "patients_df":       patients,
        "lang_dummies":      lang_dummies,
    }


# ═══════════════════════════════════════════════════════════════════
# LOO PREDICTION
# ═══════════════════════════════════════════════════════════════════

def _loo_predict(pipe, X: np.ndarray, y: np.ndarray,
                 class_names: list[str],
                 lang_dummies: np.ndarray | None = None):
    """Produce leave-one-out predictions and class probabilities for a pipeline.
    In each fold, language is residualised out of the features using dummies fit
    on the training split only (no leakage), the pipeline is cloned and fit on
    the training fold, and the held-out point is predicted. Returns the predicted
    labels and the per-class probability matrix aligned to class_names."""
    n      = len(y)
    y_pred = np.empty(n, dtype=object)
    y_prob = np.zeros((n, len(class_names)))

    for train_idx, test_idx in LeaveOneOut().split(X):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()

        if lang_dummies is not None:
            C_tr = lang_dummies[train_idx]
            C_te = lang_dummies[test_idx]
            for i in range(X_train.shape[1]):
                col   = X_train[:, i]
                valid = ~np.isnan(col)
                if valid.sum() < 5:
                    continue
                lr = LinearRegression().fit(C_tr[valid], col[valid])
                X_train[valid, i] = col[valid] - lr.predict(C_tr[valid])
                X_test[:, i]      = X_test[:, i] - lr.predict(C_te)

        p = clone(pipe)
        p.fit(X_train, y[train_idx])
        y_pred[test_idx] = p.predict(X_test)
        proba    = p.predict_proba(X_test)
        cls_list = list(p.named_steps["clf"].classes_)
        for j, c in enumerate(class_names):
            if c in cls_list:
                y_prob[test_idx, j] = proba[:, cls_list.index(c)]

    return y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════
# LOO-ALIGNED PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def _loo_permutation_importance(
        pipe,
        X: np.ndarray,
        y: np.ndarray,
        lang_dummies: np.ndarray | None = None,
        n_repeats: int = N_PERM_REPEATS,
        random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """LOO-aligned permutation importance. For each fold, fit on the training
    split, then for each feature replace the held-out point's value with
    n_repeats draws from the training distribution and measure the drop in
    prediction correctness. Returns (importances_mean, importances_std) over
    folds. Drawing from the training distribution is the LOO analogue of
    permutation importance: it breaks the feature-label link at the test point
    without touching the training data."""
    rng      = np.random.default_rng(random_state)
    n, n_feat = X.shape
    # fold_imps[fold, feat] = importance for that fold/feature
    fold_imps = np.full((n, n_feat), np.nan)

    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X)):
        X_train = X[train_idx].copy()
        X_test  = X[test_idx].copy()   # shape (1, n_feat)
        y_train = y[train_idx]
        y_test  = y[test_idx]          # shape (1,)

        # ── Language residualisation (fit on train only) ──────────
        if lang_dummies is not None:
            C_tr = lang_dummies[train_idx]
            C_te = lang_dummies[test_idx]
            for i in range(n_feat):
                col   = X_train[:, i]
                valid = ~np.isnan(col)
                if valid.sum() < 5:
                    continue
                lr = LinearRegression().fit(C_tr[valid], col[valid])
                X_train[valid, i] = col[valid] - lr.predict(C_tr[valid])
                X_test[:, i]      = X_test[:, i] - lr.predict(C_te)

        # ── Fit model on training fold ────────────────────────────
        p = clone(pipe)
        p.fit(X_train, y_train)

        # Baseline: correct (1) or wrong (0) on the held-out point
        baseline_correct = float(p.predict(X_test)[0] == y_test[0])

        # ── Per-feature permutation ───────────────────────────────
        for j in range(n_feat):
            # Draw n_repeats values from the training distribution of feat j
            train_vals = X_train[:, j]
            train_vals = train_vals[~np.isnan(train_vals)]
            if len(train_vals) == 0:
                fold_imps[fold, j] = 0.0
                continue

            perm_correct = np.empty(n_repeats)
            for r in range(n_repeats):
                X_perm    = X_test.copy()
                X_perm[0, j] = rng.choice(train_vals)
                perm_correct[r] = float(p.predict(X_perm)[0] == y_test[0])

            # Positive importance = model does worse when feature is permuted
            fold_imps[fold, j] = baseline_correct - perm_correct.mean()

    importances_mean = np.nanmean(fold_imps, axis=0)
    importances_std  = np.nanstd(fold_imps,  axis=0)
    return importances_mean, importances_std

# ═══════════════════════════════════════════════════════════════════
# LOO-ALIGNED SHAP IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def _loo_shap_importance(
        pipe,
        X: np.ndarray,
        y: np.ndarray,
        feats: list[str],
        lang_dummies: np.ndarray | None = None,
) -> pd.DataFrame:
    """LOO-aligned SHAP importance for linear models (LR or linear SVM). Per
    fold: residualise on language (train only), fit, then compute LinearExplainer
    SHAP values for the held-out point with the training set as background,
    averaging |SHAP| across classes for multiclass. Returns a DataFrame
    (feature, shap_importance, shap_importance_std) sorted descending, with
    importance = mean |SHAP| across folds. SHAP values are comparable within but
    not across model types (log-odds for LR, decision function for SVM)."""

    n, n_feat = X.shape
    fold_shaps = np.full((n, n_feat), np.nan)

    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X)):
        X_train = X[train_idx].copy()
        X_test  = X[test_idx].copy()

        # ── Language residualisation (fit on train only) ──────────
        if lang_dummies is not None:
            C_tr = lang_dummies[train_idx]
            C_te = lang_dummies[test_idx]
            for i in range(n_feat):
                col   = X_train[:, i]
                valid = ~np.isnan(col)
                if valid.sum() < 5:
                    continue
                lr = LinearRegression().fit(C_tr[valid], col[valid])
                X_train[valid, i] = col[valid] - lr.predict(C_tr[valid])
                X_test[:, i]      = X_test[:, i] - lr.predict(C_te)

        # ── Fit pipeline on training fold ─────────────────────────
        p = clone(pipe)
        p.fit(X_train, y[train_idx])

        scaler = p.named_steps["scaler"]
        clf    = p.named_steps["clf"]
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        # ── SHAP via LinearExplainer ──────────────────────────────
        # masker=X_tr_s uses the training fold as background distribution,
        # consistent with how permutation importance draws from training dist.
        explainer = shap.LinearExplainer(clf, X_tr_s)
        shap_vals = explainer.shap_values(X_te_s)

        # shap_vals shape varies by shap version and problem type:
        #   multiclass (new):  (n_samples, n_features, n_classes)  ndarray
        #   multiclass (old):  list of n_classes arrays, each (n_samples, n_features)
        #   binary:            (n_samples, n_features)
        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                # (n_samples, n_features, n_classes) → mean |SHAP| across classes
                arr = np.mean(np.abs(shap_vals[0]), axis=-1)   # → (n_features,)
            elif shap_vals.ndim == 2:
                # (n_samples, n_features) — binary
                arr = np.abs(shap_vals[0])                     # → (n_features,)
            else:
                arr = np.abs(shap_vals)
        elif isinstance(shap_vals, list):
            # old shap: list of (n_samples, n_features) arrays, one per class
            arr = np.mean([np.abs(sv[0]) for sv in shap_vals], axis=0)  # → (n_features,)
        else:
            arr = np.abs(shap_vals)

        fold_shaps[fold] = arr

    mean_shap = np.nanmean(fold_shaps, axis=0)
    std_shap  = np.nanstd(fold_shaps,  axis=0)

    return pd.DataFrame({
        "feature":              feats,
        "shap_importance":      mean_shap.round(4),
        "shap_importance_std":  std_shap.round(4),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP AUC CI
# ═══════════════════════════════════════════════════════════════════

def _bootstrap_auc_ci(y_enc, y_prob, n_class):
    """Bootstrap a 95% confidence interval for the ROC-AUC on the LOO prediction
    vector. Resamples patients with replacement N_BOOTSTRAP times, recomputing
    binary or OVR macro AUC each time and skipping degenerate resamples, then
    returns the 2.5th and 97.5th percentiles (NaN, NaN if none are valid)."""
    rng     = np.random.default_rng(RANDOM_STATE)
    n       = len(y_enc)
    samples = []
    for _ in range(N_BOOTSTRAP):
        idx    = rng.integers(0, n, size=n)
        yt, yp = y_enc[idx], y_prob[idx]
        if len(np.unique(yt)) < n_class:
            continue
        try:
            if n_class == 2:
                pos = yp[:, 1]
                if np.unique(pos).size < 2:
                    continue
                samples.append(roc_auc_score(yt, pos))
            else:
                samples.append(
                    roc_auc_score(yt, yp, multi_class="ovr", average="macro"))
        except ValueError:
            continue
    if not samples:
        return float("nan"), float("nan")
    arr = np.array(samples)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════

def _evaluate(X, y, label, n_features, problem="3way",
              lang_dummies=None) -> list[dict]:
    """Evaluate every configured classifier on one problem (3-way or a named
    binary contrast) using LOO predictions. Computes balanced accuracy, macro F1,
    Cohen's kappa, ROC-AUC with a bootstrap CI, PR-AUC against its chance level,
    and per-class recall/precision. Returns one result dict per classifier."""
    class_names = (
        [s for s in SYMPTOM_STATE_ORDER if s in np.unique(y)]
        if problem == "3way"
        else sorted(np.unique(y).tolist())
    )
    y_enc         = LabelEncoder().fit(class_names).transform(y)
    chance        = CHANCE_3WAY if problem == "3way" else CHANCE_BINARY
    class_counts  = np.bincount(y_enc)
    pr_auc_chance = float(np.mean(class_counts / len(y_enc)))

    rows = []
    for clf_name, pipe in CLASSIFIERS.items():
        is_dummy       = clf_name.startswith("Dummy")
        y_pred, y_prob = _loo_predict(pipe, X, y, class_names,
                                      lang_dummies=lang_dummies)
        bal = balanced_accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro", zero_division=0)
        kap = cohen_kappa_score(y, y_pred)

        try:
            if problem == "3way":
                auc = roc_auc_score(y_enc, y_prob,
                                    multi_class="ovr", average="macro")
            else:
                pos = y_prob[:, 1]
                auc = (roc_auc_score(y_enc, pos)
                       if np.unique(pos).size > 1 else float("nan"))
        except ValueError:
            auc = float("nan")

        try:
            pr_auc = (average_precision_score(y_enc, y_prob, average="macro")
                      if problem == "3way"
                      else average_precision_score(y_enc, y_prob[:, 1]))
        except ValueError:
            pr_auc = float("nan")

        ci_lo, ci_hi = _bootstrap_auc_ci(y_enc, y_prob, len(class_names))
        cm        = confusion_matrix(y, y_pred, labels=class_names)
        per_class = {}
        for i, cls in enumerate(class_names):
            tp  = cm[i, i]
            fn  = cm[i, :].sum() - tp
            fp  = cm[:, i].sum() - tp
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            pre = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            per_class[f"recall_{cls}"]    = round(rec, 4)
            per_class[f"precision_{cls}"] = round(pre, 4)

        above = bool(not is_dummy and not np.isnan(ci_lo) and ci_lo > 0.5)
        row = {
            "label":               label,
            "problem":             problem,
            "classifier":          clf_name,
            "n":                   int(len(y)),
            "n_features":          int(n_features),
            "balanced_accuracy":   round(bal, 4),
            "f1_macro":            round(f1m, 4),
            "auc_macro":           round(auc, 4) if not np.isnan(auc) else None,
            "auc_boot_lo":         round(ci_lo, 4) if not np.isnan(ci_lo) else None,
            "auc_boot_hi":         round(ci_hi, 4) if not np.isnan(ci_hi) else None,
            "cohen_kappa":         round(kap, 4),
            "below_chance":        bool(bal < chance),
            "auc_above_chance":    above,
            "pr_auc":              round(pr_auc, 4) if not np.isnan(pr_auc) else None,
            "pr_auc_chance":       round(pr_auc_chance, 4),
            "pr_auc_above_chance": bool(
                not is_dummy and not np.isnan(pr_auc) and pr_auc > pr_auc_chance),
        }
        row.update(per_class)
        rows.append(row)
    return rows


def _print_rows(rows):
    """Pretty-print a list of evaluation result rows to the console, showing the
    headline metrics per classifier and flagging below-chance results, AUC CIs
    that clear 0.5, and cases where probabilities look poorly calibrated."""
    for r in rows:
        auc = f"{r['auc_macro']:.3f}" if r["auc_macro"] is not None else "—"
        pr  = f"{r['pr_auc']:.3f}"    if r.get("pr_auc") is not None else "—"
        ci  = (f"boot=[{r['auc_boot_lo']:.3f},{r['auc_boot_hi']:.3f}]"
               if r["auc_boot_lo"] is not None else "boot=[—,—]")
        pr_chance  = r.get("pr_auc_chance", 0.333)
        pr_val     = r.get("pr_auc") or 0
        calib_warn = (not r["classifier"].startswith("Dummy")
                      and r["balanced_accuracy"] > 0.55
                      and pr_val < pr_chance + 0.05)
        flag = ("  ⚠ BELOW CHANCE" if r["below_chance"]
                else "  ✓ CI>0.5"  if r.get("auc_above_chance") else "")
        if calib_warn and not r["below_chance"]:
            flag += "  ⚠ prob.uncalib"
        print(f"  {r['classifier']:<22} "
              f"BalAcc={r['balanced_accuracy']:.3f}  "
              f"F1={r['f1_macro']:.3f}  "
              f"κ={r['cohen_kappa']:.3f}  "
              f"ROC-AUC={auc} {ci}  "
              f"PR-AUC={pr}(chance={pr_chance:.3f}){flag}")


# ═══════════════════════════════════════════════════════════════════
# WORD-COUNT BASELINE
# ═══════════════════════════════════════════════════════════════════

def run_word_count_baseline(m):
    """Evaluate a reference classifier that uses only word count as a feature,
    for the 3-way problem. Returns the evaluation rows (tagged as the
    word_count_baseline set), or an empty list if word_count is unavailable."""
    pdf = m["patients_df"]
    if "word_count" not in pdf.columns:
        print("\n  ⚠ word_count baseline skipped — column not found")
        return []
    X_wc = pdf[["word_count"]].values.astype(float)
    y_wc = m["symptom_state"]
    print(f"\n[word_count_baseline]  n_features=1  n={len(y_wc)}")
    rs = _evaluate(X_wc, y_wc, label="word_count_baseline",
                   n_features=1, problem="3way",
                   lang_dummies=m.get("lang_dummies"))
    for r in rs:
        r["set"] = "word_count_baseline"
    _print_rows(rs)
    return rs


# ═══════════════════════════════════════════════════════════════════
# 3-WAY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def run_3way(m):
    """Run the 3-way mania/euthymia/depression classification: the word-count
    baseline, the full SGA-feature evaluation, and permutation plus SHAP feature
    importance. Prints the top features and returns the result rows together with
    the permutation and SHAP importance frames."""
    print("\n" + "=" * 70)
    print("3-WAY CLASSIFICATION  (mania / euthymia / depression)")
    print("=" * 70)

    feats = m["selected_features"]
    X     = m["X"][feats].values.astype(float)
    y     = m["symptom_state"]

    rows = run_word_count_baseline(m)

    print(f"\n[speech_graphs_emotion]  n_features={len(feats)}  n={len(y)}")
    rs = _evaluate(X, y, label="speech_graphs_emotion",
                   n_features=len(feats), problem="3way",
                   lang_dummies=m.get("lang_dummies"))
    for r in rs:
        r["set"] = "speech_graphs_emotion"
    _print_rows(rs)
    rows.extend(rs)

    perm_df, shap_df = _importance_3way(m)

    print("\n  Permutation importance (3-way, LR):")
    for _, row in perm_df[perm_df["classifier"] == "LogisticRegression"] \
                      .sort_values("importance", ascending=False).head(6).iterrows():
        print(f"    {row['feature']:<12} {row['importance']:+.4f} "
              f"+/- {row['importance_std']:.4f}")

    print("\n  SHAP importance (3-way, LR):")
    for _, row in shap_df[shap_df["classifier"] == "LogisticRegression"] \
                      .head(6).iterrows():
        print(f"    {row['feature']:<12} {row['shap_importance']:+.4f} "
              f"+/- {row['shap_importance_std']:.4f}")

    return rows, perm_df, shap_df


def _importance_3way(m) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Permutation importance (LR + SVM) and SHAP importance (LR + SVM)
    for the 3-way problem. Returns (perm_df, shap_df).
    """
    feats        = m["selected_features"]
    X            = m["X"][feats].values.astype(float)
    y            = m["symptom_state"]
    lang_dummies = m.get("lang_dummies")

    perm_rows = []
    shap_rows = []

    for clf_name in ("LogisticRegression", "SVM_linear"):
        pipe = clone(CLASSIFIERS[clf_name])

        # ── Permutation ───────────────────────────────────────────
        imp_mean, imp_std = _loo_permutation_importance(
            pipe, X, y,
            lang_dummies=lang_dummies,
            n_repeats=N_PERM_REPEATS,
            random_state=RANDOM_STATE,
        )
        for feat, imp, std in zip(feats, imp_mean, imp_std):
            perm_rows.append({
                "classifier":     clf_name,
                "feature":        feat,
                "importance":     round(float(imp), 4),
                "importance_std": round(float(std), 4),
            })

        # ── SHAP ─────────────────────────────────────────────────
        shap_df = _loo_shap_importance(pipe, X, y, feats,
                                       lang_dummies=lang_dummies)
        shap_df.insert(0, "classifier", clf_name)
        shap_rows.append(shap_df)

    perm_df = pd.DataFrame(perm_rows)
    shap_df = pd.concat(shap_rows, ignore_index=True)
    return perm_df, shap_df


# ═══════════════════════════════════════════════════════════════════
# BINARY CONTRASTS
# ═══════════════════════════════════════════════════════════════════

def run_binary_contrasts(m):
    """Run the binary classification contrasts: active-vs-euthymia plus every
    pairwise mood contrast that has at least five patients per side. Each contrast
    is evaluated with the same LOO pipeline and per-fold language residualisation.
    Returns the combined evaluation rows across all contrasts."""
    print("\n" + "=" * 70)
    print("BINARY CONTRASTS")
    print("=" * 70)

    feats        = m["selected_features"]
    X_df         = m["X"].copy()
    X_df["symptom_state"] = m["symptom_state"]
    lang_dummies = m.get("lang_dummies")
    all_rows     = []

    df_bin = X_df.copy()
    df_bin["binary_state"] = np.where(
        df_bin["symptom_state"] == "euthymia", "euthymia", "active")
    y_bin = df_bin["binary_state"].values
    n_eu  = int((y_bin == "euthymia").sum())
    n_ac  = int((y_bin == "active").sum())
    if min(n_eu, n_ac) >= 5:
        X_bin = df_bin[feats].values.astype(float)
        print(f"\n[active_vs_euthymia]  n_active={n_ac}  n_euth={n_eu}")
        rs = _evaluate(X_bin, y_bin,
                       label="speech_graphs_emotion",
                       n_features=len(feats),
                       problem="active_vs_euthymia",
                       lang_dummies=lang_dummies)
        for r in rs:
            r["contrast"] = "active_vs_euthymia"
        _print_rows(rs)
        all_rows.extend(rs)

    states = [s for s in SYMPTOM_STATE_ORDER
              if s in X_df["symptom_state"].unique()]
    for s1, s2 in combinations(states, 2):
        mask = X_df["symptom_state"].isin([s1, s2])
        sub  = X_df[mask].copy()
        y_pw = sub["symptom_state"].values
        n1   = int((y_pw == s1).sum())
        n2   = int((y_pw == s2).sum())
        if min(n1, n2) < 5:
            continue
        X_pw    = sub[feats].values.astype(float)
        lang_pw = lang_dummies[sub.index.tolist()] if lang_dummies is not None else None
        label   = f"{s1}_vs_{s2}"
        print(f"\n[{label}]  n_{s1}={n1}  n_{s2}={n2}")
        rs = _evaluate(X_pw, y_pw,
                       label="speech_graphs_emotion",
                       n_features=len(feats),
                       problem=label,
                       lang_dummies=lang_pw)
        for r in rs:
            r["contrast"] = label
        _print_rows(rs)
        all_rows.extend(rs)

    return all_rows


# ═══════════════════════════════════════════════════════════════════
# BINARY FEATURE INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

def extract_binary_lr_coefficients(m) -> pd.DataFrame:
    """
    LOO-averaged coefficients for directional interpretation.
    For each LOO fold: residualise on train, fit on train, extract coef.
    Mean and SD are computed across all n folds.
    Sign-adjusted so positive = higher values predict the positive class.
    """
    feats        = m["selected_features"]
    X_df         = m["X"].copy()
    X_df["symptom_state"] = m["symptom_state"]
    lang_dummies = m.get("lang_dummies")
    all_rows     = []

    for clf_name in ("LogisticRegression", "SVM_linear"):
        for (s1, s2), pos_cls in CONTRAST_POSITIVE_CLASS.items():
            neg_cls = s2 if pos_cls == s1 else s1
            mask    = X_df["symptom_state"].isin([s1, s2])
            sub     = X_df[mask].copy()
            y       = sub["symptom_state"].values
            X       = sub[feats].values.astype(float)

            if len(np.unique(y)) < 2 or min(
                    (y == s1).sum(), (y == s2).sum()) < 5:
                continue

            lang_sub = (lang_dummies[sub.index.tolist()]
                        if lang_dummies is not None else None)

            # ── LOO coefficient extraction ────────────────────────
            fold_coefs = np.full((len(y), len(feats)), np.nan)

            for fold, (train_idx, test_idx) in enumerate(
                    LeaveOneOut().split(X)):
                X_train = X[train_idx].copy()
                y_train = y[train_idx]

                # Residualise on train only
                if lang_sub is not None:
                    C_tr = lang_sub[train_idx]
                    for i in range(X_train.shape[1]):
                        col   = X_train[:, i]
                        valid = ~np.isnan(col)
                        if valid.sum() < 5:
                            continue
                        lr = LinearRegression().fit(C_tr[valid], col[valid])
                        X_train[valid, i] = col[valid] - lr.predict(C_tr[valid])

                pipe = clone(CLASSIFIERS[clf_name])
                pipe.fit(X_train, y_train)
                clf  = pipe.named_steps["clf"]

                coef = clf.coef_[0].copy()
                if clf.classes_[1] != pos_cls:
                    coef = -coef

                fold_coefs[fold] = coef

            mean_coef = np.nanmean(fold_coefs, axis=0)
            std_coef  = np.nanstd(fold_coefs,  axis=0)

            for j, feat in enumerate(feats):
                all_rows.append({
                    "classifier":       clf_name,
                    "contrast":         f"{s1}_vs_{s2}",
                    "positive_class":   pos_cls,
                    "negative_class":   neg_cls,
                    "feature":          feat,
                    "coefficient":      round(float(mean_coef[j]), 4),
                    "coefficient_std":  round(float(std_coef[j]),  4),
                })

    return pd.DataFrame(all_rows)

def binary_importance(m) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Permutation importance (LR + SVM) and SHAP importance (LR + SVM)
    for each binary contrast. Returns (perm_df, shap_df).
    """
    feats        = m["selected_features"]
    X_df         = m["X"].copy()
    X_df["symptom_state"] = m["symptom_state"]
    lang_dummies = m.get("lang_dummies")

    perm_rows = []
    shap_rows = []

    for (s1, s2) in CONTRAST_POSITIVE_CLASS.keys():
        mask = X_df["symptom_state"].isin([s1, s2])
        sub  = X_df[mask].copy()
        y    = sub["symptom_state"].values
        X    = sub[feats].values.astype(float)

        if len(np.unique(y)) < 2 or min(
                (y == s1).sum(), (y == s2).sum()) < 5:
            continue

        lang_sub = (lang_dummies[sub.index.tolist()]
                    if lang_dummies is not None else None)
        contrast = f"{s1}_vs_{s2}"

        for clf_name in ("LogisticRegression", "SVM_linear"):
            pipe = clone(CLASSIFIERS[clf_name])

            # ── Permutation ───────────────────────────────────────
            print(f"  Perm importance [{clf_name}]: {contrast} (n={len(y)}) …")
            imp_mean, imp_std = _loo_permutation_importance(
                pipe, X, y,
                lang_dummies=lang_sub,
                n_repeats=N_PERM_REPEATS,
                random_state=RANDOM_STATE,
            )
            for feat, imp, std in zip(feats, imp_mean, imp_std):
                perm_rows.append({
                    "contrast":       contrast,
                    "classifier":     clf_name,
                    "feature":        feat,
                    "importance":     round(float(imp), 4),
                    "importance_std": round(float(std), 4),
                })

            # ── SHAP ─────────────────────────────────────────────
            print(f"  SHAP importance [{clf_name}]: {contrast} …")
            sdf = _loo_shap_importance(pipe, X, y, feats,
                                       lang_dummies=lang_sub)
            sdf.insert(0, "classifier", clf_name)
            sdf.insert(0, "contrast",   contrast)
            shap_rows.append(sdf)

    perm_df = pd.DataFrame(perm_rows)
    shap_df = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    return perm_df, shap_df


# ═══════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """Recursively convert numpy scalars and arrays inside nested dicts/lists into
    native Python types so the results structure can be serialised with json."""
    if isinstance(obj, dict):          return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_classification():
    """Entry point for the speech-graph emotion classification analysis. Loads the
    data, runs the 3-way and binary problems with their permutation/SHAP
    importances and binary LR coefficients, and writes all result CSVs plus a JSON
    summary to OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("MOOD STATE CLASSIFICATION")
    print("Methodology: Speech Graphs  |  Task: Emotion")
    print("CV: LOO  |  Zero-variance filter  |  7 SGA metrics")
    print("Language: one-hot dummies residualised per LOO fold")
    print("Permutation importance: LOO-aligned (draws from training dist.)")
    print("=" * 70)

    m = load_speech_graphs_emotion()
    if m is None:
        print("\n✗ Failed to load data — aborting.")
        return

    rows_3way, perm_3way, shap_3way = run_3way(m)
    rows_bin                         = run_binary_contrasts(m)
    coef_df                          = extract_binary_lr_coefficients(m)

    print("\n── Binary importance (permutation + SHAP, LR + SVM) ──")
    perm_bin, shap_bin = binary_importance(m)

    ALL_STATE_COLS = [f"{metric}_{s}"
                      for metric in ["recall", "precision"]
                      for s in ["mania", "euthymia", "depression", "active"]]

    df_3way = pd.DataFrame(rows_3way)
    for col in ALL_STATE_COLS:
        if col not in df_3way.columns:
            df_3way[col] = None
    df_3way.to_csv(OUTPUT_DIR / "classification_3way.csv", index=False)

    df_bin = pd.DataFrame(rows_bin)
    for col in ALL_STATE_COLS:
        if col not in df_bin.columns:
            df_bin[col] = None
    df_bin.to_csv(OUTPUT_DIR / "classification_binary.csv", index=False)

    perm_3way.to_csv(OUTPUT_DIR / "classification_importance_3way.csv",        index=False)
    shap_3way.to_csv(OUTPUT_DIR / "classification_shap_importance_3way.csv",   index=False)
    perm_bin.to_csv( OUTPUT_DIR / "classification_importance_binary.csv",       index=False)
    shap_bin.to_csv( OUTPUT_DIR / "classification_shap_importance_binary.csv",  index=False)

    if not coef_df.empty:
        coef_df.to_csv(OUTPUT_DIR / "classification_binary_coefficients.csv", index=False)


    results = {
        "methodology": "speech_graphs",
        "task":        TASK,
        "selected_features": m["selected_features"],
        "n_patients":  len(m["caliber_code"]),
        "mood_state_counts": {
            s: int((m["symptom_state"] == s).sum())
            for s in SYMPTOM_STATE_ORDER
        },
        "config": {
            "chance_3way":          CHANCE_3WAY,
            "chance_binary":        CHANCE_BINARY,
            "n_bootstrap":          N_BOOTSTRAP,
            "n_perm_repeats":       N_PERM_REPEATS,
            "random_state":         RANDOM_STATE,
            "cv":                   "LeaveOneOut",
            "feature_selection":    "7 SGA metrics, zero-variance filter",
            "language_control":     "one-hot dummies (drop_first=True), residualised per fold",
            "classifiers":          list(CLASSIFIERS.keys()),
            "auc_chance_note":      "OVR macro-AUC chance = 0.5 (not 1/3)",
            "bootstrap_ci_note":    "CI on LOO prediction vector only",
            "importance_method":    (
                "LOO-aligned permutation: test point feature replaced by "
                f"{N_PERM_REPEATS} draws from training distribution per fold"
            ),
        },
        "3way_results":   rows_3way,
        "binary_results": rows_bin,
    }
    with (OUTPUT_DIR / "classification_results.json").open("w") as f:
        json.dump(_json_safe(results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved → {OUTPUT_DIR}/")
    print(f"  classification_3way.csv")
    print(f"  classification_binary.csv")
    print(f"  classification_importance_3way.csv")
    print(f"  classification_shap_importance_3way.csv")
    print(f"  classification_importance_binary.csv")
    print(f"  classification_shap_importance_binary.csv")
    print(f"  classification_binary_coefficients.csv")
    print(f"  classification_results.json")


if __name__ == "__main__":
    run_classification()