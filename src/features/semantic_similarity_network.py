"""
Semantic similarity networks.

Builds per-transcript networks where nodes are words (or semantic windows) and
edges connect nodes whose embeddings exceed a cosine-similarity threshold, then
extracts topology features. Embeddings come from three sources: XLM-RoBERTa
(contextual, per-word), fastText (static, per-word, language-specific), and
pre-computed sentence-transformer window embeddings. Runs at multiple thresholds
(0.3/0.5/0.7). Cross-sectional mode uses AUDIO_DATA_UNIQUE and can additionally
run a stopword-removal experiment; longitudinal mode uses AUDIO_DATA_LONGITUDINAL
and writes a per-session CSV consumed by the longitudinal analysis.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import torch
from scipy.stats import skew as _skew
from transformers import XLMRobertaModel, XLMRobertaTokenizer

warnings.filterwarnings("ignore")
from src.config import (
    AUDIO_DATA_UNIQUE,
    AUDIO_DATA_LONGITUDINAL,
    DATA_PROCESSED,
    FASTTEXT_DEFAULT_LANG,
    FASTTEXT_LANG_MODELS,
    RESULTS_DIR,
    TRANSCRIPT_PART_CLINICAL_EMBEDDING,
    TRANSCRIPT_PART_EMOTION_EMBEDDING,
    XLM_ROBERTA_DIR,
)

from src.stats_helpers import (
    analyze_experiment,
    consistency_across_experiments,
    fold_mixed,
    print_consistency_summary,
    print_experiment_summary,
)
from src.utils import read_text_with_fallback as _read_text
from src.utils import clean_caliber_code_str as _clean_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_NAME = "semantic_networks"
OUT_DIR    = Path(RESULTS_DIR) / LEVEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

LONG_OUT_DIR = Path(RESULTS_DIR) / "longitudinal" / LEVEL_NAME

TOPIC_EMBEDDING_DIR        = Path(DATA_PROCESSED) / "topic_embeddings"
SENTENCE_TRANSFORMER_SLUGS = ["mpnet", "minilm", "e5_large", "labse"]

# Similarity thresholds for edge creation
SIMILARITY_THRESHOLDS = [0.3, 0.5, 0.7]

# Minimum tokens/nodes for valid network
MIN_TOKENS = 10

# Tasks
TASK_DIRS = {
    "emotion":             TRANSCRIPT_PART_EMOTION_EMBEDDING,
    "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_EMBEDDING,
}

# Language mapping
LANG_TO_FT: dict[str, str] = {
    "spanish": "es",
    "catalan": "ca",
    "english": "en",
    "mix":     FASTTEXT_DEFAULT_LANG,
}

# Network topology features
TOPOLOGY_FEATURES = [
    "n_nodes",
    "n_edges",
    "density",
    "mean_degree",
    "std_degree",
    "max_degree",
    "mean_clustering",
    "std_clustering",
    "transitivity",
    "mean_pagerank",
    "std_pagerank",
    "assortativity",
    "n_components",
    "largest_component_size",
    "diameter",
    "mean_shortest_path",
    "global_efficiency",
    "modularity",
    "mean_edge_weight",
    "std_edge_weight",
    "median_edge_weight",
]


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD FILTERING
# ══════════════════════════════════════════════════════════════════════════════

# Pronouns kept because they carry clinical information
_PRONOUNS = {
    # English
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    # Spanish
    "yo", "mi", "mí", "me", "nosotros", "nosotras", "nuestro", "nuestra",
    "tú", "te", "ti", "tu", "usted", "él", "ella", "ellos", "ellas",
    "su", "sus", "le", "lo", "la", "los", "las", "se", "sí",
    # Catalan
    "jo", "em", "nosaltres", "nostre", "nostra",
    "tu", "et", "vostè", "ell", "ella", "ells", "elles",
    "seu", "seva", "seus", "seves", "es",
}

# High-frequency function words removed when stopword filtering is on
_STOPWORDS = {
    # Spanish articles
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    # Spanish conjunctions
    'y', 'e', 'o', 'u', 'pero', 'mas', 'sino', 'que', 'porque', 'pues',
    'como', 'cuando', 'donde', 'aunque', 'si', 'ni',
    # Spanish prepositions
    'en', 'de', 'a', 'con', 'por', 'para', 'sin', 'sobre', 'tras',
    'desde', 'hasta', 'hacia', 'entre', 'durante', 'mediante',
    # Spanish auxiliary/copula
    'es', 'está', 'son', 'están', 'hay', 'ser', 'estar', 'haber',
    'era', 'fueron', 'fue', 'sido', 'siendo',
    # Spanish discourse markers
    'pues', 'bueno', 'vale', 'entonces', 'ahora', 'así', 'además',
    'también', 'tampoco', 'sólo', 'solo', 'ya', 'aún', 'aun',
    # Spanish quantifiers/determiners
    'todo', 'toda', 'todos', 'todas', 'mucho', 'mucha', 'muchos', 'muchas',
    'poco', 'poca', 'pocos', 'pocas', 'más', 'menos', 'muy', 'tan',
    'tanto', 'tanta', 'otros', 'otras', 'otro', 'otra',
    # Spanish negation (NOT pronouns)
    'no', 'nada', 'nadie', 'ninguno', 'ninguna',

    # Catalan articles
    'els', 'les', 'uns', 'unes',
    # Catalan conjunctions
    'i', 'però', 'perquè', 'com', 'quan', 'on', 'encara', 'que',
    # Catalan prepositions
    'amb', 'per', 'sense', 'cap', 'des', 'fins', 'cap',
    # Catalan auxiliary/copula
    'és', 'són', 'hi', 'ha', 'haver', 'ser', 'estar',
    # Catalan discourse markers
    'doncs', 'aleshores', 'així', 'també', 'tampoc', 'ja',
    # Catalan quantifiers/determiners
    'tot', 'tota', 'tots', 'totes', 'molt', 'molta', 'molts', 'moltes',
    'poc', 'poca', 'pocs', 'poques', 'més', 'menys', 'tan',
    'altre', 'altra', 'altres',
    # Catalan negation
    'res', 'ningú',

    # English articles
    'the', 'a', 'an',
    # English conjunctions
    'and', 'or', 'but', 'if', 'because', 'as', 'when', 'where', 'while',
    # English prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'about', 'into', 'through', 'during', 'before', 'after',
    # English auxiliary/copula
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    # English discourse markers
    'so', 'then', 'now', 'well', 'also', 'too', 'just', 'only', 'very',
    # English quantifiers/determiners
    'all', 'some', 'any', 'much', 'many', 'more', 'most', 'other',
    # English negation
    'not', 'no', 'nothing', 'nobody', 'none',
}


def filter_stopwords(text: str, remove_stopwords: bool = False) -> str:
    """Remove high-frequency function words (keeping pronouns) if
    remove_stopwords is True; otherwise return text unchanged."""
    if not remove_stopwords:
        return text

    words = text.lower().split()
    filtered = []

    for word in words:
        clean_word = word.strip('.,;:!?¿¡"\'()[]{}')
        if clean_word in _PRONOUNS or clean_word not in _STOPWORDS:
            filtered.append(word)

    return ' '.join(filtered)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for an (n_words, dim) array."""
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms > 0, norms, 1.0)
    sim_matrix = normalized @ normalized.T
    return sim_matrix.clip(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_semantic_network(
    unique_words: list[str],
    word_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict:
    """Build a similarity network (edges where cosine >= threshold) and return
    its topology features."""
    import networkx as nx

    n_words = len(unique_words)
    if n_words < 2:
        return {feat: np.nan for feat in TOPOLOGY_FEATURES}

    embeddings = np.array([word_embeddings[w] for w in unique_words])
    sim_matrix = _pairwise_cosine_matrix(embeddings)

    G            = nx.Graph()
    edge_weights = []
    G.add_nodes_from(unique_words)

    for i in range(n_words):
        for j in range(i + 1, n_words):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                G.add_edge(unique_words[i], unique_words[j], weight=sim)
                edge_weights.append(sim)

    return extract_network_features(G, edge_weights)


def extract_network_features(G, edge_weights: list[float]) -> dict:
    """Extract topology features from network."""
    import networkx as nx

    n = G.number_of_nodes()
    m = G.number_of_edges()

    features = {
        "n_nodes":  n,
        "n_edges":  m,
        "density":  nx.density(G),
    }

    if n == 0:
        return {feat: np.nan for feat in TOPOLOGY_FEATURES}

    # Degree statistics
    degrees = np.array([d for _, d in G.degree()])
    features.update({
        "mean_degree": float(np.mean(degrees)),
        "std_degree":  float(np.std(degrees)),
        "max_degree":  int(np.max(degrees)),
    })

    # Clustering
    clustering = np.array(list(nx.clustering(G).values()))
    features.update({
        "mean_clustering": float(np.mean(clustering)),
        "std_clustering":  float(np.std(clustering)),
        "transitivity":    nx.transitivity(G),
    })

    # PageRank
    try:
        pagerank = np.array(list(nx.pagerank(G, weight="weight").values()))
        features.update({
            "mean_pagerank": float(np.mean(pagerank)),
            "std_pagerank":  float(np.std(pagerank)),
        })
    except Exception:
        features.update({"mean_pagerank": np.nan, "std_pagerank": np.nan})

    # Assortativity
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        features["assortativity"] = np.nan

    # Components
    components = list(nx.connected_components(G))
    features.update({
        "n_components":           len(components),
        "largest_component_size": len(max(components, key=len)) if components else 0,
    })

    # Path-based metrics (on largest component)
    if components:
        lcc = G.subgraph(max(components, key=len)).copy()
        if len(lcc) > 1:
            try:
                features["diameter"]           = nx.diameter(lcc)
                features["mean_shortest_path"] = nx.average_shortest_path_length(lcc)
            except Exception:
                features["diameter"]           = np.nan
                features["mean_shortest_path"] = np.nan
        else:
            features["diameter"]           = 0
            features["mean_shortest_path"] = 0
    else:
        features["diameter"]           = np.nan
        features["mean_shortest_path"] = np.nan

    # Global efficiency
    features["global_efficiency"] = nx.global_efficiency(G)

    # Modularity
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight="weight")
        features["modularity"] = nx.community.modularity(G, comms, weight="weight")
    except Exception:
        features["modularity"] = np.nan

    # Edge weight statistics
    if edge_weights:
        features.update({
            "mean_edge_weight":   float(np.mean(edge_weights)),
            "std_edge_weight":    float(np.std(edge_weights)),
            "median_edge_weight": float(np.median(edge_weights)),
        })
    else:
        features.update({
            "mean_edge_weight":   np.nan,
            "std_edge_weight":    np.nan,
            "median_edge_weight": np.nan,
        })

    return features


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

# ── XLM-RoBERTa ───────────────────────────────────────────────────────────────

_xlm_tokenizer = None
_xlm_model     = None


def _load_xlm() -> None:
    """Load the XLM-RoBERTa tokenizer and model once, caching them in module
    globals; subsequent calls are no-ops."""
    global _xlm_tokenizer, _xlm_model
    if _xlm_tokenizer is None:
        log.info("Loading XLM-RoBERTa...")
        _xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_ROBERTA_DIR)
        _xlm_model     = XLMRobertaModel.from_pretrained(XLM_ROBERTA_DIR)
        _xlm_model.eval()


def get_xlm_word_embeddings(text: str) -> dict[str, np.ndarray]:
    """Per-word XLM-RoBERTa embeddings, averaged across all contexts a word
    appears in. Returns {word: mean_embedding}."""
    _load_xlm()

    words = text.lower().split()
    if not words:
        return {}

    enc      = _xlm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    word_ids = _xlm_tokenizer(text, truncation=True, max_length=512).word_ids()

    with torch.no_grad():
        hidden = _xlm_model(**enc).last_hidden_state.squeeze(0).numpy()

    word_to_vecs: dict[int, list] = defaultdict(list)
    for tok_idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            word_to_vecs[word_idx].append(hidden[tok_idx])

    word_embeddings = {}
    for word_idx, vecs in word_to_vecs.items():
        if word_idx < len(words):
            word_embeddings[words[word_idx]] = np.mean(vecs, axis=0)

    # Average context-specific embeddings for duplicate words
    final_embeddings = {}
    for word in set(words):
        word_vecs = [word_embeddings[w] for w in words if w == word and w in word_embeddings]
        if word_vecs:
            final_embeddings[word] = np.mean(word_vecs, axis=0)

    return final_embeddings


# ── fastText ───────────────────────────────────────────────────────────────────

def get_fasttext_word_embeddings(
    text: str, ft_model
) -> tuple[dict[str, np.ndarray], float]:
    """Per-word fastText embeddings. Returns ({word: embedding}, oov_rate)."""
    words = text.lower().split()
    if not words:
        return {}, 1.0

    word_embeddings = {}
    oov_count       = 0

    for word in set(words):
        vec  = ft_model.get_word_vector(word)
        norm = np.linalg.norm(vec)
        if norm > 0:
            word_embeddings[word] = vec
        else:
            oov_count += 1

    oov_rate = oov_count / len(set(words))
    return word_embeddings, oov_rate


# ── Sentence Transformers ──────────────────────────────────────────────────────

def get_sentence_transformer_embeddings(
    task: str, slug: str, code: str
) -> tuple[dict[str, np.ndarray], list[str]] | None:
    """Load pre-computed sentence-level window embeddings.
    Returns ({window_id: embedding}, [window_texts]) or None if unavailable."""
    emb_dir  = TOPIC_EMBEDDING_DIR / slug / task
    emb_path = emb_dir / f"{code}_emb.npy"
    txt_path = emb_dir / f"{code}_win.txt"

    if not emb_path.exists() or not txt_path.exists():
        return None

    embeddings = np.load(emb_path).astype(np.float32)
    windows    = txt_path.read_text(encoding="utf-8").splitlines()

    if embeddings.shape[0] != len(windows):
        return None

    word_embeddings = {f"window_{i}": embeddings[i] for i in range(len(windows))}
    return word_embeddings, windows


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def build_xlm_features(
    task: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    remove_stopwords: bool = False,
    longitudinal: bool = False,
) -> dict[float, pd.DataFrame]:
    """Build semantic networks from XLM-RoBERTa embeddings for one task, per
    threshold. Returns {threshold: DataFrame}. In longitudinal mode fold_mixed()
    is skipped."""
    suffix = " (no stopwords)" if remove_stopwords else ""
    mode   = " [LONG]" if longitudinal else ""
    log.info(f"[XLM-RoBERTa{suffix}{mode}] Processing task: {task}")
    _load_xlm()

    transcript_dir = Path(TASK_DIRS[task])
    if not transcript_dir.exists():
        log.warning(f"Transcript dir not found: {transcript_dir}")
        return {}

    valid_codes = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())

    rows_by_threshold: dict[float, list] = {t: [] for t in SIMILARITY_THRESHOLDS}
    skipped             = 0
    skipped_not_in_csv  = 0

    for fpath in sorted(transcript_dir.glob("*.txt")):
        code = _clean_code(fpath.stem, task)

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        text_raw = _read_text(fpath)
        if not text_raw:
            skipped += 1
            continue

        text             = filter_stopwords(text_raw, remove_stopwords)
        word_embeddings  = get_xlm_word_embeddings(text)

        if len(word_embeddings) < MIN_TOKENS:
            skipped += 1
            continue

        unique_words    = list(word_embeddings.keys())
        word_count      = len(text.split())
        word_count_raw  = len(text_raw.split())

        for threshold in SIMILARITY_THRESHOLDS:
            features      = build_semantic_network(unique_words, word_embeddings, threshold)
            source_suffix = "_nostop" if remove_stopwords else ""
            row = {
                "caliber_code":    code,
                "word_count":      word_count,
                "word_count_raw":  word_count_raw,
                "n_unique_words":  len(unique_words),
                "source":          f"xlm_t{int(threshold*100)}{source_suffix}",
                "task":            task,
                "threshold":       threshold,
                "stopwords_removed": remove_stopwords,
            }
            row.update(features)
            rows_by_threshold[threshold].append(row)

    if skipped_not_in_csv:
        log.info(f"  Skipped {skipped_not_in_csv} files not in clinical data")
    if skipped:
        log.info(f"  Skipped {skipped} transcripts (too short / no embeddings)")

    out: dict[float, pd.DataFrame] = {}
    for threshold, rows in rows_by_threshold.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
        n_before = len(df)
        df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
        if not longitudinal:
            df = fold_mixed(df)
        n_after = len(df)

        if n_after < n_before:
            log.info(f"  ⚠ Dropped {n_before - n_after} rows at threshold {threshold}")

        source_suffix = "_nostop" if remove_stopwords else ""
        log.info(f"  [xlm t={threshold:.2f}{source_suffix}{mode}] {len(df)} rows")
        out[threshold] = df

    return out


def build_fasttext_features(
    task: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    remove_stopwords: bool = False,
    longitudinal: bool = False,
) -> tuple[dict[float, pd.DataFrame], list[dict]]:
    """Build semantic networks from fastText embeddings for one task, per
    threshold. Returns ({threshold: DataFrame}, oov_log). In longitudinal mode
    fold_mixed() is skipped."""
    suffix = " (no stopwords)" if remove_stopwords else ""
    mode   = " [LONG]" if longitudinal else ""
    log.info(f"[fastText{suffix}{mode}] Processing task: {task}")

    transcript_dir = Path(TASK_DIRS[task])
    if not transcript_dir.exists():
        log.warning(f"Transcript dir not found: {transcript_dir}")
        return {}, []

    valid_codes = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())

    lang_lookup: dict[str, str] = {}
    if "language" in df_clinical.columns:
        lang_lookup = (
            df_clinical.set_index("caliber_code")["language"]
            .dropna().str.lower().str.strip().to_dict()
        )

    files_by_lang: dict[str, list[Path]] = {}
    skipped_not_in_csv = 0

    for fpath in sorted(transcript_dir.glob("*.txt")):
        code = _clean_code(fpath.stem, task)

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        raw_lang = lang_lookup.get(code)
        lang_key = LANG_TO_FT.get(str(raw_lang or "").strip().lower(), FASTTEXT_DEFAULT_LANG)
        files_by_lang.setdefault(lang_key, []).append(fpath)

    if skipped_not_in_csv:
        log.info(f"  Skipped {skipped_not_in_csv} files not in clinical data")

    rows_by_threshold: dict[float, list] = {t: [] for t in SIMILARITY_THRESHOLDS}
    oov_log: list[dict] = []
    skipped             = 0

    for lang_key, fpaths in files_by_lang.items():
        log.info(f"  Processing {len(fpaths)} files in {lang_key}...")

        path     = FASTTEXT_LANG_MODELS.get(lang_key, FASTTEXT_LANG_MODELS[FASTTEXT_DEFAULT_LANG])
        log.info(f"    Loading fastText [{lang_key}] from {path}...")
        ft_model = fasttext.load_model(str(path))

        for fpath in fpaths:
            code     = _clean_code(fpath.stem, task)
            text_raw = _read_text(fpath)
            if not text_raw:
                skipped += 1
                continue

            text                         = filter_stopwords(text_raw, remove_stopwords)
            word_embeddings, oov_rate    = get_fasttext_word_embeddings(text, ft_model)
            patient_lang                 = lang_lookup.get(code)

            oov_log.append({
                "caliber_code":      code,
                "task":              task,
                "language":          patient_lang,
                "fasttext_lang":     lang_key,
                "oov_rate":          oov_rate,
                "word_count":        len(text.split()),
                "word_count_raw":    len(text_raw.split()),
                "n_unique_words":    len(word_embeddings),
                "stopwords_removed": remove_stopwords,
            })

            if len(word_embeddings) < MIN_TOKENS:
                skipped += 1
                continue

            unique_words   = list(word_embeddings.keys())
            word_count     = len(text.split())
            word_count_raw = len(text_raw.split())

            for threshold in SIMILARITY_THRESHOLDS:
                features      = build_semantic_network(unique_words, word_embeddings, threshold)
                source_suffix = "_nostop" if remove_stopwords else ""
                row = {
                    "caliber_code":      code,
                    "word_count":        word_count,
                    "word_count_raw":    word_count_raw,
                    "n_unique_words":    len(unique_words),
                    "oov_rate":          oov_rate,
                    "source":            f"fasttext_t{int(threshold*100)}{source_suffix}",
                    "task":              task,
                    "threshold":         threshold,
                    "stopwords_removed": remove_stopwords,
                }
                row.update(features)
                rows_by_threshold[threshold].append(row)

        del ft_model
        gc.collect()
        log.info(f"    Cleared {lang_key} model from memory")

    if skipped:
        log.info(f"  Skipped {skipped} transcripts (too short)")

    out: dict[float, pd.DataFrame] = {}
    for threshold, rows in rows_by_threshold.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
        n_before = len(df)
        df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
        if not longitudinal:
            df = fold_mixed(df)
        n_after = len(df)

        if n_after < n_before:
            log.info(f"  ⚠ Dropped {n_before - n_after} rows at threshold {threshold}")

        source_suffix = "_nostop" if remove_stopwords else ""
        log.info(f"  [fasttext t={threshold:.2f}{source_suffix}{mode}] "
                 f"{len(df)} rows, OOV avg={df['oov_rate'].mean():.2f}")
        out[threshold] = df

    return out, oov_log


def build_sentence_transformer_features(
    task: str,
    slug: str,
    df_clinical: pd.DataFrame,
    clinical_cols: list[str],
    longitudinal: bool = False,
) -> dict[float, pd.DataFrame]:
    """Build semantic networks from pre-computed sentence-transformer window
    embeddings for one model (slug) and task, per threshold. Returns
    {threshold: DataFrame}. In longitudinal mode the pre-computed embeddings must
    contain the visit-specific caliber_codes; if none match the result is empty
    and a warning is printed."""
    mode = " [LONG]" if longitudinal else ""
    log.info(f"[{slug}{mode}] Processing task: {task}")

    emb_dir     = TOPIC_EMBEDDING_DIR / slug / task
    if not emb_dir.is_dir():
        log.warning(f"  Embedding dir not found: {emb_dir}")
        return {}

    valid_codes        = set(df_clinical["caliber_code"].astype(str).str.strip().str.lower())
    rows_by_threshold: dict[float, list] = {t: [] for t in SIMILARITY_THRESHOLDS}
    skipped_not_in_csv = 0
    skipped            = 0

    for emb_path in sorted(emb_dir.glob("*_emb.npy")):
        code = emb_path.stem.replace("_emb", "")

        if code not in valid_codes:
            skipped_not_in_csv += 1
            continue

        result = get_sentence_transformer_embeddings(task, slug, code)
        if result is None:
            skipped += 1
            continue

        word_embeddings, windows = result

        if len(word_embeddings) < 2:
            skipped += 1
            continue

        unique_words = list(word_embeddings.keys())
        word_count   = sum(len(w.split()) for w in windows)

        for threshold in SIMILARITY_THRESHOLDS:
            features = build_semantic_network(unique_words, word_embeddings, threshold)
            row = {
                "caliber_code":      code,
                "word_count":        word_count,
                "n_windows":         len(windows),
                "source":            f"{slug}_t{int(threshold*100)}",
                "task":              task,
                "threshold":         threshold,
                "stopwords_removed": False,
            }
            row.update(features)
            rows_by_threshold[threshold].append(row)

    if skipped_not_in_csv:
        entity = "longitudinal sessions" if longitudinal else "patients"
        log.info(f"  Skipped {skipped_not_in_csv} embeddings not matching {entity}")
    if skipped:
        log.info(f"  Skipped {skipped} patients (insufficient windows)")

    out: dict[float, pd.DataFrame] = {}
    for threshold, rows in rows_by_threshold.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
        n_before = len(df)
        df = df.merge(df_clinical[clinical_cols], on="caliber_code", how="inner")
        if not longitudinal:
            df = fold_mixed(df)
        n_after = len(df)

        if n_after < n_before:
            log.info(f"  ⚠ Dropped {n_before - n_after} rows at threshold {threshold}")

        log.info(f"  [{slug} t={threshold:.2f}{mode}] {len(df)} rows")
        out[threshold] = df

    if not any(v is not None and (not isinstance(v, pd.DataFrame) or not v.empty) for v in out.values()) and longitudinal:
        log.warning(
            f"  ⚠ [{slug}] {task}: no longitudinal session codes found in "
            f"pre-computed embeddings. Re-run the embedding stage with "
            f"longitudinal codes to populate Source 3 for the longitudinal subset."
        )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# LONGITUDINAL SESSION CSV
# ══════════════════════════════════════════════════════════════════════════════

def _build_longitudinal_sessions_csv(
    patient_rows: list[pd.DataFrame],
    out_dir: Path,
) -> None:
    """Aggregate per-session feature rows into the sessions CSV consumed by the
    longitudinal analysis: one row per (caliber_code, source_family, task) with
    TOPOLOGY_FEATURES averaged across thresholds. source_family strips the
    threshold suffix (e.g. xlm_t30/t50/t70 -> xlm) and is renamed back to
    'source' on output."""
    if not patient_rows:
        log.warning("  ⚠ No longitudinal session rows to write.")
        return

    df = pd.concat(patient_rows, ignore_index=True)

    # Derive source family: strip _t<N> threshold suffix and _nostop suffix
    def _family(src: str) -> str:
        src = re.sub(r"_t\d+", "", src)     # strip threshold suffix
        src = re.sub(r"_nostop$", "", src)  # strip stopword flag
        return src

    df["source_family"] = df["source"].apply(_family)

    # Columns that identify a unique session × source combination
    non_feature_cols = {"source", "source_family", "threshold", "n_unique_words",
                        "n_windows", "oov_rate", "word_count_raw", "stopwords_removed"}
    id_cols          = ["caliber_code", "task", "source_family"]
    clinical_pass    = [c for c in df.columns
                        if c not in non_feature_cols
                        and c not in TOPOLOGY_FEATURES
                        and c not in id_cols
                        and c != "word_count"]

    agg_cols = TOPOLOGY_FEATURES + (["word_count"] if "word_count" in df.columns else [])
    agg_cols = [c for c in agg_cols if c in df.columns]

    agg_df = (
        df.groupby(id_cols + clinical_pass, as_index=False)[agg_cols]
        .mean()
    )

    agg_df = agg_df.rename(columns={"source_family": "source"})

    out_path = out_dir / "semantic_networks_sessions.csv"
    agg_df.to_csv(out_path, index=False)
    log.info(f"\n  ✓ Longitudinal sessions CSV → {out_path}")
    log.info(f"    {len(agg_df)} rows  ·  "
             f"{agg_df['caliber_code'].nunique()} unique session codes  ·  "
             f"{agg_df['source'].nunique()} sources")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_semantic_network_analysis(
    tasks: list[str] | None = None,
    use_xlm: bool = True,
    use_fasttext: bool = True,
    use_sentence_transformers: bool = True,
    run_stopword_experiment: bool = False,
    longitudinal: bool = False,
) -> None:
    """Run the semantic similarity network pipeline over the given embedding
    sources and tasks. Cross-sectional mode uses AUDIO_DATA_UNIQUE (optionally
    running the paired stopword experiment); longitudinal mode uses
    AUDIO_DATA_LONGITUDINAL, writes to LONG_OUT_DIR, and produces a sessions CSV."""
    tasks = tasks or list(TASK_DIRS.keys())

    if longitudinal:
        run_stopword_experiment = False   # keep runtime manageable
        out_dir      = LONG_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        clinical_csv = AUDIO_DATA_LONGITUDINAL
        mode_label   = "LONGITUDINAL"
    else:
        out_dir      = OUT_DIR
        clinical_csv = AUDIO_DATA_UNIQUE
        mode_label   = "CROSS-SECTIONAL"

    file_suffix = "_stopword_exp" if run_stopword_experiment else ""

    log.info("=" * 70)
    log.info(f"  LEVEL 3B — SEMANTIC SIMILARITY NETWORKS  [{mode_label}]")
    log.info("  Nodes: words/windows")
    log.info("  Edges: semantic similarity above threshold")
    log.info(f"  Thresholds: {SIMILARITY_THRESHOLDS}")

    if longitudinal:
        log.info("\n  ⚠  LONGITUDINAL MODE")
        log.info("     • Clinical data: AUDIO_DATA_LONGITUDINAL")
        log.info("     • Outputs:       results/longitudinal/semantic_networks/")
        log.info("     • Stopword experiment disabled (run cross-sectional pipeline separately)")
        log.info("     • fold_mixed skipped (s11 handles this via _fold_mixed helper)")
        log.info("     • Cross-experiment consistency skipped (handled by s11)")

    if run_stopword_experiment:
        log.info("\n  🧪 STOPWORD EXPERIMENT: Running BOTH with and without function word filtering")
        log.info("     Hypothesis: XLM floor effect driven by high-frequency function words")

    log.info("=" * 70)

    # ── Load clinical data ────────────────────────────────────────────────────
    df_clinical = pd.read_csv(clinical_csv)
    df_clinical["caliber_code"] = (
        df_clinical["caliber_code"].astype(str).str.strip().str.lower()
    )
    clinical_cols = [
        c for c in ["caliber_code", "symptom_state", "language", "sex", "age", "ymrs", "hdrs"]
        if c in df_clinical.columns
    ]

    all_results:  list[pd.DataFrame] = []
    patient_rows: list[pd.DataFrame] = []
    all_oov_log:  list[dict]         = []

    filter_conditions = [False]
    if run_stopword_experiment:
        filter_conditions.append(True)

    # ── Source 1: XLM-RoBERTa ─────────────────────────────────────────────────
    if use_xlm:
        for remove_sw in filter_conditions:
            suffix = " [STOPWORD EXPERIMENT]" if remove_sw else ""
            log.info("\n" + "─" * 60)
            log.info(f"  SOURCE 1: XLM-RoBERTa (contextual, token-level){suffix}")
            log.info("─" * 60)

            for task in tasks:
                dfs_by_threshold = build_xlm_features(
                    task, df_clinical, clinical_cols,
                    remove_stopwords=remove_sw,
                    longitudinal=longitudinal,
                )

                for threshold, df in dfs_by_threshold.items():
                    if df.empty:
                        continue

                    patient_rows.append(df)

                    if not longitudinal:
                        metric_cols = [
                            c for c in TOPOLOGY_FEATURES
                            if c in df.columns and df[c].nunique() > 1
                        ]
                        if len(df) >= 10 and metric_cols:
                            exp_suffix = "_nostop" if remove_sw else ""
                            exp_name   = f"xlm_t{int(threshold*100)}{exp_suffix}_{task}"
                            res        = analyze_experiment(df, metric_cols)
                            if not res.empty:
                                res.insert(0, "experiment", exp_name)
                                res["source"]            = f"xlm_t{int(threshold*100)}{exp_suffix}"
                                res["task"]              = task
                                res["threshold"]         = threshold
                                res["stopwords_removed"] = remove_sw
                                all_results.append(res)
                                print_experiment_summary(exp_name, res)

    # ── Source 2: fastText ────────────────────────────────────────────────────
    if use_fasttext:
        for remove_sw in filter_conditions:
            suffix = " [STOPWORD EXPERIMENT]" if remove_sw else ""
            log.info("\n" + "─" * 60)
            log.info(f"  SOURCE 2: fastText (static, word-level){suffix}")
            log.info("─" * 60)

            for task in tasks:
                dfs_by_threshold, oov_log = build_fasttext_features(
                    task, df_clinical, clinical_cols,
                    remove_stopwords=remove_sw,
                    longitudinal=longitudinal,
                )
                all_oov_log.extend(oov_log)

                for threshold, df in dfs_by_threshold.items():
                    if df.empty:
                        continue

                    patient_rows.append(df)

                    if not longitudinal:
                        metric_cols = [
                            c for c in TOPOLOGY_FEATURES
                            if c in df.columns and df[c].nunique() > 1
                        ]
                        if len(df) >= 10 and metric_cols:
                            exp_suffix = "_nostop" if remove_sw else ""
                            exp_name   = f"fasttext_t{int(threshold*100)}{exp_suffix}_{task}"
                            res        = analyze_experiment(df, metric_cols)
                            if not res.empty:
                                res.insert(0, "experiment", exp_name)
                                res["source"]            = f"fasttext_t{int(threshold*100)}{exp_suffix}"
                                res["task"]              = task
                                res["threshold"]         = threshold
                                res["stopwords_removed"] = remove_sw
                                all_results.append(res)
                                print_experiment_summary(exp_name, res)

    # ── Source 3: Sentence Transformers ───────────────────────────────────────
    if use_sentence_transformers:
        log.info("\n" + "─" * 60)
        log.info("  SOURCE 3: Sentence Transformers (contextual, sentence-level)")
        if longitudinal:
            log.info("  ⚠  Pre-computed embeddings must contain session-level codes.")
        log.info("─" * 60)

        for slug in SENTENCE_TRANSFORMER_SLUGS:
            slug_dir = TOPIC_EMBEDDING_DIR / slug
            if not slug_dir.is_dir():
                log.warning(f"  {slug} not found - run Level 6 first")
                continue

            for task in tasks:
                dfs_by_threshold = build_sentence_transformer_features(
                    task, slug, df_clinical, clinical_cols,
                    longitudinal=longitudinal,
                )

                for threshold, df in dfs_by_threshold.items():
                    if df.empty:
                        continue

                    patient_rows.append(df)

                    if not longitudinal:
                        metric_cols = [
                            c for c in TOPOLOGY_FEATURES
                            if c in df.columns and df[c].nunique() > 1
                        ]
                        if len(df) >= 10 and metric_cols:
                            exp_name = f"{slug}_t{int(threshold*100)}_{task}"
                            res      = analyze_experiment(df, metric_cols)
                            if not res.empty:
                                res.insert(0, "experiment", exp_name)
                                res["source"]            = f"{slug}_t{int(threshold*100)}"
                                res["task"]              = task
                                res["threshold"]         = threshold
                                res["stopwords_removed"] = False
                                all_results.append(res)
                                print_experiment_summary(exp_name, res)

    # ══════════════════════════════════════════════════════════════════════════
    # LONGITUDINAL: write sessions CSV and exit
    # ══════════════════════════════════════════════════════════════════════════
    if longitudinal:
        _build_longitudinal_sessions_csv(patient_rows, out_dir)

        if all_oov_log:
            pd.DataFrame(all_oov_log).to_csv(
                out_dir / "fasttext_oov_log_longitudinal.csv", index=False)
            log.info(f"  fastText OOV log → {out_dir / 'fasttext_oov_log_longitudinal.csv'}")

        log.info(f"\n{'═'*70}")
        log.info("  LONGITUDINAL SEMANTIC NETWORK EXTRACTION COMPLETE")
        log.info(f"  Outputs → {out_dir}")
        log.info(f"{'═'*70}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-SECTIONAL: consistency + stopword analysis
    # ══════════════════════════════════════════════════════════════════════════
    if not all_results:
        log.warning("\n✗ No experiments produced results.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(out_dir / f"semantic_networks_per_experiment{file_suffix}.csv", index=False)

    # ── Stopword experiment analysis ──────────────────────────────────────────
    if run_stopword_experiment and patient_rows:
        log.info("\n" + "═" * 70)
        log.info("  STOPWORD FILTERING EXPERIMENT RESULTS")
        log.info("═" * 70)

        patient_df     = pd.concat(patient_rows, ignore_index=True)
        xlm_patient_df = patient_df[
            patient_df["source"].str.contains("xlm", case=False)
        ].copy()

        if not xlm_patient_df.empty and "stopwords_removed" in xlm_patient_df.columns:
            log.info("\n📊 XLM-RoBERTa Networks: Impact of Stopword Removal")
            log.info("─" * 70)

            key_metrics = ["mean_edge_weight", "density", "mean_clustering", "transitivity"]

            for metric in key_metrics:
                if metric not in xlm_patient_df.columns:
                    continue

                with_stop = xlm_patient_df[~xlm_patient_df["stopwords_removed"]]
                no_stop   = xlm_patient_df[xlm_patient_df["stopwords_removed"]]

                if len(with_stop) > 0 and len(no_stop) > 0:
                    mean_with = with_stop[metric].mean()
                    mean_no   = no_stop[metric].mean()
                    std_with  = with_stop[metric].std()
                    std_no    = no_stop[metric].std()

                    log.info(f"\n  {metric}:")
                    log.info(f"    With stopwords:    {mean_with:.4f} (SD={std_with:.4f})")
                    log.info(f"    Without stopwords: {mean_no:.4f} (SD={std_no:.4f})")
                    log.info(f"    Δ (change):        {mean_no - mean_with:+.4f}")

                    if metric == "mean_edge_weight":
                        if mean_no < mean_with - 0.01:
                            log.info(f"    ✓ Mean edge weight DECREASED (more differentiation)")
                        elif mean_no > mean_with + 0.01:
                            log.info(f"    ✗ Mean edge weight INCREASED (compression worsened)")
                        else:
                            log.info(f"    ≈ No substantial change (<1%)")

            comparison_rows = []
            for task in xlm_patient_df["task"].unique():
                for threshold in xlm_patient_df["threshold"].unique():
                    base_source  = f"xlm_t{int(threshold*100)}"
                    mask_with    = (
                        (xlm_patient_df["task"] == task) &
                        (xlm_patient_df["threshold"] == threshold) &
                        (~xlm_patient_df["stopwords_removed"])
                    )
                    mask_without = (
                        (xlm_patient_df["task"] == task) &
                        (xlm_patient_df["threshold"] == threshold) &
                        (xlm_patient_df["stopwords_removed"])
                    )
                    with_data    = xlm_patient_df[mask_with]
                    without_data = xlm_patient_df[mask_without]

                    if len(with_data) > 0 and len(without_data) > 0:
                        row = {
                            "source":               base_source,
                            "task":                 task,
                            "threshold":            threshold,
                            "n_patients_with":      len(with_data),
                            "n_patients_without":   len(without_data),
                        }
                        for metric in key_metrics:
                            if metric in with_data.columns:
                                row[f"{metric}_with"]    = with_data[metric].mean()
                                row[f"{metric}_without"] = without_data[metric].mean()
                                row[f"{metric}_delta"]   = (
                                    without_data[metric].mean() - with_data[metric].mean()
                                )
                        comparison_rows.append(row)

            if comparison_rows:
                stopword_comparison = pd.DataFrame(comparison_rows)
                stopword_comparison.to_csv(
                    out_dir / f"semantic_networks_stopword_comparison{file_suffix}.csv",
                    index=False,
                )
                log.info(f"\n  Stopword comparison → "
                         f"{out_dir / f'semantic_networks_stopword_comparison{file_suffix}.csv'}")

                log.info("\n  Summary by threshold (emotion task):")
                log.info("  " + "─" * 66)
                for _, row in stopword_comparison.iterrows():
                    if row["task"] == "emotion":
                        log.info(
                            f"  {row['source']:15s}  "
                            f"edge_weight: {row.get('mean_edge_weight_with', 0):.4f} → "
                            f"{row.get('mean_edge_weight_without', 0):.4f}  "
                            f"(Δ {row.get('mean_edge_weight_delta', 0):+.4f})"
                        )

    # ── Cross-experiment consistency ───────────────────────────────────────────
    log.info("\n" + "═" * 70)
    log.info("  CROSS-EXPERIMENT CONSISTENCY")
    log.info("═" * 70)

    consistency_df = consistency_across_experiments(results_df)
    print_consistency_summary(consistency_df)
    consistency_df.to_csv(
        out_dir / f"semantic_networks_consistency{file_suffix}.csv", index=False)

    # ── Patient-level output ───────────────────────────────────────────────────
    if patient_rows:
        patient_df = pd.concat(patient_rows, ignore_index=True)
        patient_df["experiment"] = (
            patient_df["source"].astype(str) + "_" +
            patient_df["task"].astype(str)
        )
        patient_df.to_csv(
            out_dir / f"semantic_networks_patient_features{file_suffix}.csv", index=False)
        log.info(f"\n  Patient features → "
                 f"{out_dir / f'semantic_networks_patient_features{file_suffix}.csv'}")
        log.info(f"    {len(patient_df)} observations, "
                 f"{patient_df['caliber_code'].nunique()} patients, "
                 f"{patient_df['experiment'].nunique()} experiments")

    # ── OOV log ────────────────────────────────────────────────────────────────
    if all_oov_log:
        oov_df = pd.DataFrame(all_oov_log)
        oov_df.to_csv(out_dir / f"fasttext_oov_log{file_suffix}.csv", index=False)
        log.info(f"  fastText OOV log → {out_dir / f'fasttext_oov_log{file_suffix}.csv'}")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info(f"\n{'═'*70}")
    log.info("  SUMMARY")
    log.info(f"{'═'*70}")
    log.info(f"  Per-experiment results: {len(results_df)} experiments × metrics")
    log.info(f"  Consistency summary:    {len(consistency_df)} features")
    log.info(f"  Patient features:       {len(patient_df) if patient_rows else 0} rows")
    if run_stopword_experiment:
        log.info(f"  Stopword comparison:    "
                 f"{out_dir / f'semantic_networks_stopword_comparison{file_suffix}.csv'}")
    log.info(f"\n  All outputs saved to: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Level 3B - Semantic Similarity Networks"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_DIRS.keys()),
        action="append",
        default=None,
        dest="tasks",
        help="Task(s) to process. Default: all.",
    )
    parser.add_argument(
        "--source",
        choices=["xlm", "fasttext", "sentence_transformers", "all"],
        action="append",
        default=None,
        dest="sources",
        help="Embedding source(s) to run. Default: all.",
    )
    parser.add_argument(
        "--stopword-experiment",
        action="store_true",
        help="Run parallel analysis with/without stopword filtering.",
    )
    parser.add_argument(
        "--longitudinal",
        action="store_true",
        help="Run in longitudinal mode: load AUDIO_DATA_LONGITUDINAL, "
             "write to results/longitudinal/semantic_networks/, "
             "produce semantic_networks_sessions.csv for s11.",
    )

    args    = parser.parse_args()
    tasks   = args.tasks or list(TASK_DIRS.keys())
    sources = set(args.sources or ["all"])
    use_all = "all" in sources

    run_semantic_network_analysis(
        tasks,
        use_xlm                   = use_all or "xlm"                   in sources,
        use_fasttext              = use_all or "fasttext"              in sources,
        use_sentence_transformers = use_all or "sentence_transformers" in sources,
        run_stopword_experiment   = args.stopword_experiment,
        longitudinal              = args.longitudinal,
    )