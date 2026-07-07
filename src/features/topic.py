"""
Topic analysis of clinical-evaluation and emotion transcripts (exploratory).

For each sentence-embedding model, windows pooled across patients are clustered
with BERTopic (UMAP + HDBSCAN, noise windows reassigned to the nearest topic).
Per patient it computes topic-distribution shape metrics (entropy, top-1,
effective-N, Simpson diversity), sequential-transition metrics (topic_switch_rate,
topic_run_length) and pct_symptom_windows (windows naming a psychiatric condition
from a curated multilingual vocabulary), plus state over-representation profiles
and a language-confounding check. A cross-model selection table ranks models by
clean topics and robust shape metrics. Inputs are the pre-computed window
embeddings and the clinical metadata CSV.

Outputs (under RESULTS_DIR/level6_topic_analysis/<model_slug>/):
  topics_top_words.csv             top words + size per topic
  topics_examples.csv              up to 3 representative windows per topic
  topics_state_profile.csv         over-representation ratio per state
  topics_state_means.csv           mean fraction per (topic × state)
  topics_language_crosstab.csv     language distribution per topic
  topics_per_experiment.csv        per-topic statistical results
  topics_shape_per_experiment.csv  shape + transition + symptom statistics
  topics_shape_patient_values.csv  patient-level shape metrics for plotting
  umap_2d_projection.csv           2-D coordinates for scatter plot
  model_selection_summary.csv      cross-model ranking (top level)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP

from src.features.topic_embeddings import _windows
from src.config import (AUDIO_DATA_UNIQUE,AUDIO_DATA_LONGITUDINAL, DATA_PROCESSED, RESULTS_DIR,
                        TRANSCRIPT_PART_CLINICAL_CLEAN, TRANSCRIPT_PART_EMOTION_CLEAN)
from src.stats_helpers import (
    analyze_experiment, fold_mixed,
    print_experiment_summary, STATES,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_NAME          = "topic_analysis"
TOPIC_EMBEDDING_DIR = Path(DATA_PROCESSED) / "topic_embeddings"


TASK_DIRS = {
    "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_CLEAN,
    "emotion":            TRANSCRIPT_PART_EMOTION_CLEAN,
}

TARGET_NR_TOPICS        = 8
N_PROFILE_TOPICS        = 5
LANG_CONFOUND_THRESHOLD = 0.85
REASSIGN_OUTLIERS       = True


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
# Curated multilingual clinical/diagnostic terms; a window counts as a "symptom
# window" if it contains >=1 of these tokens (whole-word, case-insensitive),
# capturing illness meta-language rather than affective content.
# Cf. Villa-Pérez et al. (2023); Girard et al. (2022).

SYMPTOM_VOCAB = frozenset("""
    depression depresion depressio depressió
    mania mania mani manía
    bipolar bipolar
    anxiety ansiedad ansietat ansiosa ansios ansiós
    schizophrenia esquizofrenia esquizofrènia
    psychosis psicosis psicosi
    suicid suïcid suicidio suïcidi
    hallucination alucinacion al·lucinació
    delusion delirio deliri
    paranoia paranoia paranoïa
    panic pànic panico pánico
    stress estrés estres estrès
    insomnia insomnio insomni
    medication medicacion medicació medicament medicamento
    psychiatrist psiquiatra psiquiatre
    therapy terapia teràpia
    treatment tractament tratamiento
    crisis crisi crisis
    relapse recaida recaiguda
    episode episodi episodio
    hospitalized hospitalizado hospitalitzat hospital
    sad triste trist tristeza tristesa
    hopeless desesperanza desesperança
    worthless inutile inútil
    tired cansat cansado
    exhausted esgotat agotado
""".split())


# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS  (c-TF-IDF top-word labelling only)
# ─────────────────────────────────────────────────────────────────────────────

ES_CA_EN_STOPWORDS = sorted(set("""
el la los las un una unos unas de del a al en con por para que como
y o pero porque cuando donde mientras aunque sin sobre entre
es son era eran fue fueron ha han había habían sea sean ser estar
está están estaba estaban ese esa eso esos esas
no si sí me mi yo lo he ella él ellos him her
jo em li va ho hi
muy mucho mucha muchos muchas
els les uns unes pel pels amb però perquè quan on mentre encara
sense és són sigui siguin està estan estava estaven
pues vale hola eh
the a an of in on with for that and or but as at by it its not so
is are was were be been has have had
um uh yeah okay
l d m s n i
""".split()))


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: str) -> str:
    """Read a text file as UTF-8, falling back to Windows-1252 on decode error."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding="windows-1252") as f:
            return f.read()


def _save(rows: list, filename: str, out_dir: Path) -> None:
    """Concatenate a list of DataFrames (or dict rows) and write it to a CSV in
    out_dir; does nothing if the list is empty."""
    if not rows:
        return
    if isinstance(rows[0], pd.DataFrame):
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(out_dir / filename, index=False)


def load_pool(task: str, df_unique: pd.DataFrame, model_root: Path
              ) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Pool per-window texts and embeddings across patients for one task.
    Returns (window_texts, embeds (n_windows, dim), patient_idx indexing df rows,
    df of per-patient clinical info)."""
    mono_dir = TASK_DIRS[task]
    emb_dir  = model_root / task
    codes, all_texts, all_embeds, all_pidx = [], [], [], []
    word_counts: dict[str, int] = {}

    for fname in sorted(os.listdir(mono_dir)):
        if not fname.endswith(".txt"):
            continue
        code = (fname.replace(f"_{task}.txt", "")
                     .replace(".txt", "").strip().lower())
        emb_path = emb_dir / f"{code}_emb.npy"
        txt_path = emb_dir / f"{code}_win.txt"
        if not (emb_path.exists() and txt_path.exists()):
            continue
        src_txt = Path(mono_dir) / fname
        wins = _windows(_read_text(str(src_txt)))
        E    = np.load(emb_path)
        if not wins or E.shape[0] != len(wins):
            continue
        word_counts[code] = len(_read_text(os.path.join(mono_dir, fname)).split())
        idx = len(codes)
        codes.append(code)
        all_texts.extend(wins)
        all_embeds.append(E)
        all_pidx.extend([idx] * len(wins))

    if not codes:
        return [], np.zeros((0, 0)), np.zeros(0, int), pd.DataFrame()

    df_files = pd.DataFrame({"caliber_code": codes})
    df_files["word_count"] = df_files["caliber_code"].map(word_counts)
    cols = [c for c in ["caliber_code", "symptom_state", "language",
                        "sex", "age", "ymrs", "hdrs"]
            if c in df_unique.columns]
    df = df_files.merge(df_unique[cols], on="caliber_code", how="inner")
    df = fold_mixed(df).reset_index(drop=True)

    keep     = set(df["caliber_code"])
    sel      = np.array([codes[i] in keep for i in all_pidx])
    embeds   = np.vstack(all_embeds)[sel]
    texts    = [t for t, m in zip(all_texts, sel) if m]
    code2new = {c: i for i, c in enumerate(df["caliber_code"])}
    pidx     = np.array(
        [code2new[codes[p]] for p, m in zip(all_pidx, sel) if m], dtype=int
    )
    return texts, embeds, pidx, df


# ─────────────────────────────────────────────────────────────────────────────
# BERTOPIC FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_bertopic(texts: list[str], embeds: np.ndarray,
                 nr_topics: int = TARGET_NR_TOPICS,
                 seed: int = 42) -> tuple[BERTopic, list[int], np.ndarray]:
    """Fit BERTopic (tuned for a ~4,000–7,000 window corpus). Returns (model,
    topic_assignments, umap_2d_projection)."""
    n = len(texts)

    n_neighbors  = min(30, max(10, n // 150))
    n_components = min(10, max(5,  n // 500))

    umap_model = UMAP(
        n_neighbors  = n_neighbors,
        n_components = n_components,
        metric       = "cosine",
        random_state = seed,
        low_memory   = False,
    )

    min_cs      = max(15, n // 50)
    min_samples = max(5,  min_cs // 3)

    hdbscan_model = HDBSCAN(
        min_cluster_size         = min_cs,
        min_samples              = min_samples,
        metric                   = "euclidean",
        cluster_selection_method = "eom",
        prediction_data          = True,
    )

    vectorizer = CountVectorizer(
        stop_words    = ES_CA_EN_STOPWORDS,
        ngram_range   = (1, 2),
        min_df        = 1,
        strip_accents = None,   # preserve accents
        token_pattern = r"(?u)\b\w[\w·-]+\b",
    )

    tm = BERTopic(
        umap_model              = umap_model,
        hdbscan_model           = hdbscan_model,
        vectorizer_model        = vectorizer,
        calculate_probabilities = False,
        verbose                 = False,
    )
    topics, _ = tm.fit_transform(texts, embeddings=embeds)
    tm.update_topics(texts, vectorizer_model=vectorizer)

    info   = tm.get_topic_info()
    n_real = (info["Topic"] != -1).sum()
    if n_real > nr_topics:
        tm.reduce_topics(texts, nr_topics=nr_topics)
        topics = list(tm.topics_)

    if REASSIGN_OUTLIERS:
        topics = _reassign_outliers(topics, tm, embeds)

    umap_2d_model = UMAP(
        n_neighbors  = n_neighbors,
        n_components = 2,
        metric       = "cosine",
        random_state = seed,
    )
    umap_2d = umap_2d_model.fit_transform(embeds)
    return tm, topics, umap_2d


def _reassign_outliers(topics: list[int], tm: BERTopic,
                       embeds: np.ndarray) -> list[int]:
    """Reassign noise windows to the nearest topic centroid (cosine distance)."""
    topic_ids = sorted(t for t in set(topics) if t != -1)
    if not topic_ids:
        return topics
    try:
        centroids = np.array([
            tm.topic_embeddings_[i + 1] for i in range(len(topic_ids))
        ], dtype=np.float32)
    except (AttributeError, IndexError):
        centroids = np.array([
            embeds[np.array(topics) == tid].mean(axis=0)
            for tid in topic_ids
        ], dtype=np.float32)

    centroids   /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    embeds_norm  = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    outlier_mask = np.array(topics) == -1
    if not outlier_mask.any():
        return topics

    sims       = embeds_norm[outlier_mask] @ centroids.T
    nearest    = np.argmax(sims, axis=1)
    topics_arr = np.array(topics, dtype=int)
    topics_arr[outlier_mask] = np.array(topic_ids)[nearest]
    return topics_arr.tolist()

def get_top_words_manual(topics, texts, topic_ids, n_words=10):
    """Derive the top c-TF-IDF words per topic by pooling each topic's windows
    into a single document and ranking terms with a TfidfVectorizer. Returns a
    {topic_id: comma-separated top words} mapping. Used as an accent-preserving
    alternative to BERTopic's built-in labelling."""
    topic_docs = {tid: [] for tid in topic_ids}
    for text, topic in zip(texts, topics):
        if topic in topic_docs:
            topic_docs[topic].append(text)
    
    corpus = [" ".join(topic_docs[tid]) for tid in topic_ids]
    
    vectorizer = TfidfVectorizer(
        stop_words    = ES_CA_EN_STOPWORDS,
        ngram_range   = (1, 2),
        min_df        = 2,
        strip_accents = None,
        token_pattern = r"(?u)\b[^\W\d_][\w·'-]*\b",
        sublinear_tf  = True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    
    result = {}
    for i, tid in enumerate(topic_ids):
        row = tfidf_matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:n_words]
        result[tid] = ", ".join(terms[j] for j in top_idx if row[j] > 0)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION & SHAPE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def patient_topic_fractions(topics: list[int], patient_idx: np.ndarray,
                            n_patients: int, topic_ids: list[int]
                            ) -> np.ndarray:
    """P[i, j] = fraction of patient i's windows in topic topic_ids[j]."""
    P       = np.zeros((n_patients, len(topic_ids)))
    counts  = np.zeros(n_patients)
    tid_pos = {tid: j for j, tid in enumerate(topic_ids)}
    for t, p in zip(topics, patient_idx):
        if t in tid_pos:
            P[p, tid_pos[t]] += 1
            counts[p]         += 1
    counts[counts == 0] = 1
    return P / counts[:, None]


def topic_distribution_metrics(P: np.ndarray,
                                topics: list[int],
                                patient_idx: np.ndarray,
                                n_patients: int,
                                topic_ids: list[int],
                                window_texts: list[str]) -> pd.DataFrame:
    """Per-patient topic-distribution shape and sequential-transition metrics:
      topic_entropy      Shannon entropy (nats) of the topic distribution.
      topic_top1         Fraction of windows in the dominant topic.
      topic_effective_n  exp(entropy), the effective number of topics used.
      topic_diversity    Simpson's diversity 1 − Σp² (increases with spread;
                         not the Gini inequality coefficient).
      topic_switch_rate  Fraction of consecutive window pairs that change topic
                         (high = loose association; low = perseveration).
      topic_run_length   Mean length (windows) of same-topic runs.
      pct_symptom_windows Fraction of windows containing >=1 SYMPTOM_VOCAB term.
    """
    P   = np.asarray(P, dtype=float)
    eps = 1e-12
    rs  = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    Pn  = P / rs

    entropy   = -np.sum(np.where(Pn > 0, Pn * np.log(Pn + eps), 0), axis=1)
    top1      = Pn.max(axis=1)
    effective = np.exp(entropy)
    diversity = 1 - np.sum(Pn ** 2, axis=1)     # Simpson's diversity index

    # ── Sequential transition metrics ────────────────────────────────────────
    switch_rates  = np.full(n_patients, np.nan)
    run_lengths   = np.full(n_patients, np.nan)

    # patient_idx preserves load_pool insertion order, so within-patient loop
    # order is the window sequence order.
    from collections import defaultdict
    pat_topics: dict[int, list[int]] = defaultdict(list)
    for t, p in zip(topics, patient_idx):
        pat_topics[p].append(t)

    for p, seq in pat_topics.items():
        if len(seq) < 2:
            continue
        transitions = [1 if seq[i] != seq[i + 1] else 0
                       for i in range(len(seq) - 1)]
        switch_rates[p] = float(np.mean(transitions))

        # Mean length of consecutive same-topic runs
        runs, run = [], 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        run_lengths[p] = float(np.mean(runs))

    # ── Symptom vocabulary: windows containing >=1 clinical/diagnostic term ────
    sym_counts = np.zeros(n_patients)
    win_counts = np.zeros(n_patients)
    for text, p in zip(window_texts, patient_idx):
        tokens = set(text.lower().split())
        sym_counts[p] += int(bool(tokens & SYMPTOM_VOCAB))
        win_counts[p] += 1
    win_counts[win_counts == 0] = 1
    pct_symptom = sym_counts / win_counts

    return pd.DataFrame({
        "topic_entropy":       entropy,
        "topic_top1":          top1,
        "topic_effective_n":   effective,
        "topic_diversity":     diversity,
        "topic_switch_rate":   switch_rates,
        "topic_run_length":    run_lengths,
        "pct_symptom_windows": pct_symptom,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STATE PROFILE  (primary exploratory output)
# ─────────────────────────────────────────────────────────────────────────────

def state_topic_profile(P: np.ndarray, states_array: np.ndarray,
                        topic_ids: list[int],
                        top_words_map: dict, examples_map: dict,
                        top_n: int = N_PROFILE_TOPICS) -> pd.DataFrame:
    """Over-representation ratio per (state, topic): topic prevalence in this
    state vs. all other states combined (>1 = over-represented). Descriptive, no
    p-value."""
    rows   = []
    states = [s for s in STATES if s in set(states_array)]
    for s in states:
        mask_in  = states_array == s
        mask_out = ~mask_in
        if mask_in.sum() == 0 or mask_out.sum() == 0:
            continue
        mean_in  = P[mask_in].mean(axis=0)
        mean_out = P[mask_out].mean(axis=0)
        ratio    = (mean_in + 1e-3) / (mean_out + 1e-3)
        ranked   = np.argsort(ratio)[::-1]
        for rank, j in enumerate(ranked[:top_n], 1):
            tid = topic_ids[j]
            rows.append({
                "symptom_state":           s,
                "rank":            rank,
                "topic":           tid,
                "mean_in_state":   float(mean_in[j]),
                "mean_other":      float(mean_out[j]),
                "over_repr_ratio": float(ratio[j]),
                "top_words":       top_words_map.get(tid, ""),
                "example_window":  examples_map.get(tid, ""),
            })
    return pd.DataFrame(rows)


def all_states_topic_summary(P: np.ndarray, states_array: np.ndarray,
                             topic_ids: list[int],
                             top_words_map: dict) -> pd.DataFrame:
    """Mean topic fraction per (state, topic) — useful for heatmap plots."""
    rows   = []
    states = [s for s in STATES if s in set(states_array)]
    for s in states:
        mask = states_array == s
        for j, tid in enumerate(topic_ids):
            vals = P[mask, j]
            rows.append({
                "symptom_state":         s,
                "topic":         tid,
                "n_patients":    int(mask.sum()),
                "mean_fraction": float(vals.mean()),
                "std_fraction":  float(vals.std()),
                "top_words":     top_words_map.get(tid, ""),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def language_topic_crosstab(topics: list[int], pidx: np.ndarray,
                            df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate each topic against the language of its windows (row-
    normalised) and flag topics whose dominant language share meets
    LANG_CONFOUND_THRESHOLD as potentially language-confounded. Returns an empty
    frame if no language column is present."""
    if "language" not in df.columns:
        return pd.DataFrame()
    lang_per_win = df["language"].iloc[pidx].values
    ct = pd.crosstab(
        pd.Series(topics, name="topic"),
        pd.Series(lang_per_win, name="language"),
        normalize="index",
    ).reset_index()
    ct["max_share"]           = ct.drop(columns="topic").max(axis=1)
    ct["language_confounded"] = ct["max_share"] >= LANG_CONFOUND_THRESHOLD
    return ct


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_model_dirs() -> list[Path]:
    """Return the embedding sub-directories under TOPIC_EMBEDDING_DIR that
    actually contain per-window embedding files for at least one task; each such
    directory corresponds to one sentence-embedding model."""
    if not TOPIC_EMBEDDING_DIR.is_dir():
        return []
    out = []
    for p in sorted(TOPIC_EMBEDDING_DIR.iterdir()):
        if not p.is_dir():
            continue
        has_data = any(
            (p / task).is_dir() and any((p / task).glob("*_emb.npy"))
            for task in TASK_DIRS
        )
        if has_data:
            out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_one_model(model_root: Path, df_unique: pd.DataFrame,
                      out_dir_root: Path) -> dict:
    """Run the full topic analysis for one embedding model across all tasks: fit
    BERTopic, label topics, run the language sanity check, compute per-patient
    topic fractions and shape/transition/symptom metrics, build state profiles,
    run statistical tests, assess cross-task consistency and write all per-model
    CSVs. Returns a summary dict of counts used for the model-selection table."""
    slug    = model_root.name
    out_dir = out_dir_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  MODEL : {slug}")
    print(f"  embeds: {model_root}")
    print("═" * 70)

    rows_topics   = []
    rows_shape    = []
    rows_shape_patient = []
    rows_state_means  = []
    rows_top_words    = []
    rows_examples     = []
    rows_lang_ct      = []
    rows_profile      = []
    rows_umap         = []

    summary = {"model": slug, "n_topics": 0, "n_confounded_topics": 0,
               "n_robust_shape_strict": 0, "n_robust_shape_pole": 0}

    for task in TASK_DIRS:
        print(f"\n── {task} ──")
        texts, embeds, pidx, df = load_pool(task, df_unique, model_root)
        if len(df) < 10:
            print(f"  ✗ only {len(df)} patients — skipping")
            continue

        cnt = df["symptom_state"].value_counts()
        print(
            f"  patients={len(df)}  windows={len(texts)}  "
            + "  ".join(f"{s}={cnt.get(s, 0)}" for s in STATES)
        )
        # ── Fit ───────────────────────────────────────────────────────────────
        tm, topics, umap_2d = fit_bertopic(texts, embeds, nr_topics=TARGET_NR_TOPICS)
        info      = tm.get_topic_info()
        topic_ids = sorted(t for t in info["Topic"] if t != -1)

        n_outliers = sum(1 for t in topics if t == -1)
        print(
            f"  topics: {len(topic_ids)}  "
            f"outlier windows after reassignment: "
            f"{n_outliers}/{len(texts)} "
            f"({100 * n_outliers / max(len(texts), 1):.1f}%)"
        )
        if not topic_ids:
            print("  ✗ no real topics found — skipping")
            continue

        # ── Interpretation aids ───────────────────────────────────────────────
        top_words_map = get_top_words_manual(topics, texts, topic_ids)
        examples_map: dict[int, str] = {}

        for tid in topic_ids:
            size = int(info.loc[info["Topic"] == tid, "Count"].iloc[0])
            rows_top_words.append({
                "task": task, "topic": tid,
                "size": size, "top_words": top_words_map.get(tid, ""),
            })
            try:
                reps = tm.get_representative_docs(tid) or []
            except Exception:
                reps = []
            if reps:
                examples_map[tid] = reps[0]
            for rank, r in enumerate(reps[:3], 1):
                rows_examples.append({
                    "task": task, "topic": tid,
                    "rank": rank, "window": r,
                })

        print("\n  Topic summary:")
        for tid in topic_ids:
            sz = int(info.loc[info["Topic"] == tid, "Count"].iloc[0])
            print(f"    topic {tid:2d} ({sz:4d} windows): {top_words_map[tid]}")

        # ── Language sanity check ─────────────────────────────────────────────
        lct = language_topic_crosstab(topics, pidx, df)
        n_conf = 0
        if not lct.empty:
            lct.insert(0, "task", task)
            rows_lang_ct.append(lct)
            n_conf = int(lct["language_confounded"].sum())
            if n_conf:
                print(
                    f"\n  ⚠  {n_conf}/{len(lct)} topics are ≥"
                    f"{LANG_CONFOUND_THRESHOLD:.0%} one language — "
                    f"interpret those topics with caution"
                )
            else:
                print("  ✓ No topics appear language-confounded")

        summary["n_topics"]            += len(topic_ids)
        summary["n_confounded_topics"] += n_conf

        # ── Per-patient topic fractions ───────────────────────────────────────
        P          = patient_topic_fractions(topics, pidx, len(df), topic_ids)
        topic_cols = [f"topic_{tid}" for tid in topic_ids]
        df         = pd.concat([df, pd.DataFrame(P, columns=topic_cols)], axis=1)

        # Compute all shape + transition + symptom metrics together
        shape_df   = topic_distribution_metrics(
            P, topics, pidx, len(df), topic_ids, texts
        )
        df         = pd.concat([df, shape_df.reset_index(drop=True)], axis=1)
        shape_cols = list(shape_df.columns)

        # ── Persist patient-level shape values for downstream plotting ────────
        cov_cols  = [c for c in ["language", "word_count", "ymrs", "hdrs", "sex", "age"]
                     if c in df.columns]
        keep_cols = ["caliber_code", "symptom_state"] + cov_cols + shape_cols
        keep_cols = [c for c in keep_cols if c in df.columns]
        pat = df[keep_cols].copy()
        pat.insert(0, "task",       task)
        pat.insert(0, "model",      slug)
        pat.insert(0, "experiment", f"{slug}_{task}")
        rows_shape_patient.append(pat)

        # ── State profile (primary exploratory output) ────────────────────────
        prof = state_topic_profile(
            P, df["symptom_state"].values,
            topic_ids, top_words_map, examples_map,
        )
        if not prof.empty:
            prof.insert(0, "task", task)
            rows_profile.append(prof)
            print("\n  State profile (top over-represented topics):")
            for s in prof["symptom_state"].unique():
                print(f"    {s}:")
                sub = prof[prof["symptom_state"] == s]
                for _, row in sub.iterrows():
                    print(
                        f"      #{int(row['rank'])} topic {int(row['topic']):2d} "
                        f"(ratio={row['over_repr_ratio']:.2f}): "
                        f"{row['top_words']}"
                    )

        sm = all_states_topic_summary(
            P, df["symptom_state"].values, topic_ids, top_words_map
        )
        sm.insert(0, "task", task)
        rows_state_means.append(sm)

        # ── Statistical tests (secondary / exploratory) ───────────────────────
        res = analyze_experiment(df, topic_cols)
        if not res.empty:
            res.insert(0, "experiment", f"{slug}_{task}")
            res.insert(1, "task",       task)
            res.insert(2, "model",      slug)
            rows_topics.append(res)
            print_experiment_summary(f"{task} [topics]", res)

        res_shape = analyze_experiment(df, shape_cols)
        if not res_shape.empty:
            res_shape.insert(0, "experiment", f"{slug}_{task}")
            res_shape.insert(1, "task",       task)
            res_shape.insert(2, "model",      slug)
            rows_shape.append(res_shape)
            print_experiment_summary(f"{task} [shape metrics]", res_shape)

    
    # ══════════════════════════════════════════════════════════════════
    # CROSS-TASK CONSISTENCY
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═'*70}\n  CROSS-TASK CONSISTENCY\n{'═'*70}")
    
    # Topic fractions (exploratory, no robustness needed)
    if rows_topics:
        df_topics_long = pd.concat(rows_topics, ignore_index=True)
        df_topics_long.to_csv(out_dir / "topics_per_experiment.csv", index=False)
        print(f"  Topic results saved (exploratory, no robustness computed)")

    # Shape metrics (confirmatory — compute robustness)
    if rows_shape:
        df_shape_long = pd.concat(rows_shape, ignore_index=True)

        from src.stats_helpers import (
            consistency_across_experiments,
            print_consistency_summary
        )

        summary_shape = consistency_across_experiments(df_shape_long)
        print("\n  Shape Metrics Robustness:")
        print_consistency_summary(summary_shape)

        summary["n_robust_shape_strict"] = int(summary_shape["robust_strict"].sum())
        summary["n_robust_shape_pole"]   = int(summary_shape["robust_pole"].sum())

        df_shape_long.to_csv(out_dir / "topics_shape_per_experiment.csv", index=False)
        summary_shape.to_csv(out_dir / "topics_shape_consistency.csv", index=False)

    # Patient-level shape values (used by main results figure)
    if rows_shape_patient:
        pd.concat(rows_shape_patient, ignore_index=True).to_csv(
            out_dir / "topics_shape_patient_values.csv", index=False
        )
    
    # ── Save per-model outputs ────────────────────────────────────────────────
    print(f"\n  Writing outputs to {out_dir}/")
    _save(rows_top_words,   "topics_top_words.csv", out_dir)
    _save(rows_examples,    "topics_examples.csv", out_dir)
    _save(rows_profile,     "topics_state_profile.csv", out_dir)
    _save(rows_state_means, "topics_state_means.csv", out_dir)
    _save(rows_lang_ct,     "topics_language_crosstab.csv", out_dir)
    _save(rows_umap,        "umap_2d_projection.csv", out_dir)
    
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_model_selection_summary(summaries: list[dict], out_dir: Path) -> None:
    """Rank the per-model summary dicts by a composite score (clean topics and
    robust shape metrics), print the ranking table with primary / cross-check
    recommendations, and save it to model_selection_summary.csv."""
    if not summaries:
        return
    df = pd.DataFrame(summaries)
    df["n_clean_topics"] = df["n_topics"] - df["n_confounded_topics"]

    df["composite_score"] = (
        df["n_clean_topics"]         * 2 +
        df["n_robust_shape_strict"]  * 5 +
        df["n_robust_shape_pole"]    * 3
    )
    
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    print("\n" + "═" * 70)
    print("  MODEL SELECTION SUMMARY")
    print("─" * 70)
    print(f"  {'Rank':<6} {'Model':<12} {'CleanTopics':<13} "
          f"{'RobustStrict':<14} {'RobustPole':<12} {'Score'}")
    for _, row in df.iterrows():
        marker = "  ← PRIMARY"     if row["rank"] == 1 else \
                 "  ← cross-check" if (row["rank"] == 2 and
                                        row["n_confounded_topics"] == 0) else ""
        print(
            f"  {int(row['rank']):<6} {row['model']:<12} "
            f"{int(row['n_clean_topics']):<13} "
            f"{int(row['n_robust_shape_strict']):<14} "
            f"{int(row['n_robust_shape_pole']):<12} "
            f"{int(row['composite_score'])}"
            f"{marker}"
        )
    print("─" * 70)
    primary = df.iloc[0]["model"]
    print(f"\n  → Use '{primary}' as primary model for thesis reporting.")
    if len(df) > 1:
        cross = df.iloc[1]
        if cross["n_confounded_topics"] == 0:
            print(f"  → Use '{cross['model']}' as multilingual cross-validation.")
    print(
        "\n  Justification: primary model selected by highest composite score\n"
        "  (clean topics × 2 + strict-robust shape × 5 + pole-robust shape × 3).\n"
        "  Robust shape metrics are those replicating across both emotion and\n"
        "  clinical evaluation tasks with consistent directional patterns.\n"
        "  Topic profiles are exploratory; shape metrics are confirmatory."
    )
    print("═" * 70)
    
    df.to_csv(out_dir / "model_selection_summary.csv", index=False)
    print(f"\n  Saved model_selection_summary.csv → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_topic_analysis(longitudinal: bool = False) -> None:
    """Entry point: discover embedding models, load the clinical metadata, run
    analyse_one_model for each model and print/save the model-selection summary.
    In longitudinal mode it reads the longitudinal clinical file and writes under
    the longitudinal results directory."""
    label = "LONGITUDINAL" if longitudinal else "CROSS-SECTIONAL"
    print("═" * 70)
    print(f" TOPIC ANALYSIS  ({label})")
    print("═" * 70)

    out_dir_root = (
        Path(RESULTS_DIR) / "longitudinal" / LEVEL_NAME
        if longitudinal
        else Path(RESULTS_DIR) / LEVEL_NAME
    )
    out_dir_root.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs()
    if not model_dirs:
        print(
            f"\n✗ No embedding model folders found under "
            f"{TOPIC_EMBEDDING_DIR}.\n  Run s11 first."
        )
        return

    print(f"  Models discovered: {[m.name for m in model_dirs]}")
    df_unique = pd.read_csv(
        AUDIO_DATA_LONGITUDINAL if longitudinal else AUDIO_DATA_UNIQUE
    )
    summaries = []
    for model_root in model_dirs:
        try:
            summary = analyse_one_model(model_root, df_unique, out_dir_root)

            summaries.append(summary)
        except Exception as exc:
            import traceback
            print(f"\n✗ {model_root.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

    print_model_selection_summary(summaries, out_dir_root)
    print(f"\n  All done.  Per-model outputs under {out_dir_root}")


if __name__ == "__main__":
    run_topic_analysis()