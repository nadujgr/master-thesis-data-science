"""
Network-specific emotional properties of word co-occurrence networks.

For each patient transcript a word co-occurrence network is built from cleaned
tokens within a sliding window, nodes are annotated with valence/arousal/
dominance (VAD) values from the NRC-VAD lexicon (English, Spanish, Catalan, plus
a merged Catalan/Spanish lexicon for bilingual patients), and network-level
emotional descriptors are computed: composition, emotional homophily (Stella et
al. 2019), Heider structural balance per VAD dimension (Teixeira et al. 2021),
emotional complexity (Stella et al. 2022) and coverage diagnostics. Inputs are
the monologue transcripts and the clinical metadata CSV; outputs are per-patient
feature CSVs, coverage diagnostics and per-/cross-experiment statistical
summaries under RESULTS_DIR/cooccurrence_emotion (or the longitudinal
equivalent). Aggregate statistics such as mean valence are left to
features/psycholinguistics.py.
"""

import logging
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from scipy.stats import skew as _skew
from statsmodels.stats.multitest import multipletests  # noqa: F401

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.config import (
    AUDIO_DATA_UNIQUE, AUDIO_DATA_LONGITUDINAL,
    FASTTEXT_LANG_MODELS,
    NRC_DIR,
    NRC_VAD_FILES,
    RESULTS_DIR,
    SPACY_MODELS as CFG_SPACY_MODELS,
    TRANSCRIPT_BY_PART_MONOLOGUE,
)
from src.stats_helpers import (
    analyze_experiment,
    consistency_across_experiments,
    fold_mixed,
    print_consistency_summary,
    print_experiment_summary,
)

RESULTS_DIR    = Path(RESULTS_DIR)
MONOLOGUE_ROOT = Path(TRANSCRIPT_BY_PART_MONOLOGUE)
CLINICAL_PATH  = Path(AUDIO_DATA_UNIQUE)
NRC_DIR        = Path(NRC_DIR)
FASTTEXT_PATHS = {k: Path(v) for k, v in FASTTEXT_LANG_MODELS.items()}
SPACY_PATHS    = {k: Path(v) for k, v in CFG_SPACY_MODELS.items()}
OUTPUT_DIR     = RESULTS_DIR / "cooccurrence_emotion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SYMPTOM_STATE_ORDER = ["mania", "depression", "euthymia"]
VAD_POS_THRESHOLD  = 0.6
VAD_NEG_THRESHOLD  = 0.4
N_BALANCE_NULL     = 1000   # following Teixeira et al. (2021)
N_HOMOPHILY_NULL   = 1000   # following Stella et al. (2019)
_ANON_RE           = re.compile(r"\[[^\]]{1,30}\]")
_VERBOSITY_PROXIES = {
    "n_nodes", "n_edges", "largest_component_size", "n_components",
}
# Label → integer for vectorised balance computation
_LABEL_INT = {"negative": -1, "low": -1, "neutral": 0, "positive": 1, "high": 1}


TOPOLOGY_BASE = [
    "n_nodes", "n_edges", "density",
    "mean_degree", "std_degree", "skew_degree", "max_degree", "assortativity",
    "mean_clustering", "std_clustering", "skew_clustering",
    "mean_pagerank", "std_pagerank", "skew_pagerank",
    "mean_betweenness", "std_betweenness", "max_betweenness",
    "mean_eigenvector", "std_eigenvector",
    "mean_closeness", "std_closeness",
    "n_components", "largest_component_size",
    "diameter", "mean_shortest_path",
    "global_efficiency", "local_efficiency", "modularity",
]

VAD_NETWORK_FEATURES = [
    # Network composition
    "pct_positive_nodes", "pct_negative_nodes", "pct_neutral_nodes",
    "pct_high_arousal_nodes", "pct_low_arousal_nodes",
    "pct_high_dominance_nodes", "pct_low_dominance_nodes",
    # Homophily (Stella et al. 2019)
    "valence_homophily", "valence_homophily_z",
    "arousal_homophily", "arousal_homophily_z",
    "dominance_homophily", "dominance_homophily_z",
    # Structural balance (Heider 1946 / Teixeira et al. 2021)
    "valence_dob", "valence_pct_neg_triads", "valence_dob_z",
    "arousal_dob", "arousal_pct_neg_triads", "arousal_dob_z",
    "dominance_dob", "dominance_pct_neg_triads", "dominance_dob_z",
    # Emotional complexity (Stella et al. 2022)
    "emotional_complexity", "emotional_coherence",
    "emotional_complexity_z_shuffled", "emotional_complexity_z_uniform",
    # LCC emotional profile
    "pct_pos_in_lcc",
    # Coverage diagnostics
    "n_nodes_total", "n_nodes_vad_covered", "pct_vad_covered",
    "n_positive_nodes", "n_negative_nodes", "n_neutral_nodes",
    "pct_nonneutral_nodes", "n_triangles", "n_valid_triads_dob",
]

EXPERIMENTS = [
    {"name": "emotion_ws2",  "task_set": "emotion",             "task_dirs": ["emotion_clean"],             "window": 2, "remove_sw": False, "note": "Emotion | WS=2."},
    {"name": "emotion_ws3",  "task_set": "emotion",             "task_dirs": ["emotion_clean"],             "window": 3, "remove_sw": False, "note": "Emotion | WS=3."},
    {"name": "emotion_ws4",  "task_set": "emotion",             "task_dirs": ["emotion_clean"],             "window": 4, "remove_sw": False, "note": "Emotion | WS=4."},
    {"name": "bdae_ws2",     "task_set": "bdae",                "task_dirs": ["bdae_clean"],                "window": 2, "remove_sw": False, "note": "BDAE | WS=2."},
    {"name": "bdae_ws3",     "task_set": "bdae",                "task_dirs": ["bdae_clean"],                "window": 3, "remove_sw": False, "note": "BDAE | WS=3."},
    {"name": "bdae_ws4",     "task_set": "bdae",                "task_dirs": ["bdae_clean"],                "window": 4, "remove_sw": False, "note": "BDAE | WS=4."},
    {"name": "clinical_ws2", "task_set": "clinical_evaluation", "task_dirs": ["clinical_evaluation_clean"], "window": 2, "remove_sw": False, "note": "Clinical | WS=2."},
    {"name": "clinical_ws3", "task_set": "clinical_evaluation", "task_dirs": ["clinical_evaluation_clean"], "window": 3, "remove_sw": False, "note": "Clinical | WS=3."},
    {"name": "clinical_ws4", "task_set": "clinical_evaluation", "task_dirs": ["clinical_evaluation_clean"], "window": 4, "remove_sw": False, "note": "Clinical | WS=4."},
]


# ── Misc helpers ───────────────────────────────────────────────────────────────
def _sep(label: str = "") -> None:
    """Print a console section separator, optionally with a short label."""
    print(f"\n  ── {label} {'─' * max(0, 52 - len(label))}" if label
          else "  " + "─" * 56)


from src.utils import json_safe as _json_safe


# ── Resource loading ───────────────────────────────────────────────────────────
def load_spacy_models() -> dict:
    """Load one spaCy pipeline per language (parser/NER disabled), falling back
    to a medium/small public model and finally a blank tokeniser if the
    configured model path is unavailable. Returns a {lang: pipeline} mapping."""
    nlp_models = {}
    fallbacks  = {"en": "en_core_web_md", "es": "es_core_news_md", "ca": "ca_core_news_sm"}
    for lang, model_path in SPACY_PATHS.items():
        try:
            if Path(model_path).exists():
                nlp_models[lang] = spacy.load(str(model_path), disable=["parser", "ner"])
                print(f"  spaCy [{lang}] loaded  ({Path(model_path).name})")
            else:
                fb = fallbacks.get(lang, lang)
                nlp_models[lang] = spacy.load(fb, disable=["parser", "ner"])
                print(f"  spaCy [{lang}] fallback '{fb}'")
        except OSError:
            print(f"  ⚠ spaCy [{lang}] not found — blank tokeniser.")
            nlp_models[lang] = spacy.blank(lang)
    return nlp_models


def load_nrc_vad(lang: str) -> dict:
    """Load the NRC-VAD lexicon for one language into a {word: {valence,
    arousal, dominance}} dictionary. Handles the differing word-column position
    across language files and averages the scores of duplicate word entries.
    Returns an empty dict if the file or the expected columns are missing."""
    path = NRC_DIR / NRC_VAD_FILES[lang]
    if not path.exists():
        print(f"  ⚠ NRC VAD not found [{lang}]: {path}"); return {}
    lines  = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].strip().split("\t")
    col_v  = next((i for i, h in enumerate(header) if h.lower() == "valence"),   None)
    col_a  = next((i for i, h in enumerate(header) if h.lower() == "arousal"),   None)
    col_d  = next((i for i, h in enumerate(header) if h.lower() == "dominance"), None)
    if None in (col_v, col_a, col_d):
        print(f"  ⚠ NRC VAD [{lang}]: V/A/D columns not found."); return {}
    word_col_idx = 0 if lang == "en" else len(header) - 1
    lexicon: dict = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(header): continue
        word = parts[word_col_idx].strip().lower()
        if not word: continue
        try: entry = {"valence": float(parts[col_v]), "arousal": float(parts[col_a]), "dominance": float(parts[col_d])}
        except (ValueError, IndexError): continue
        if word in lexicon:
            for k in ("valence", "arousal", "dominance"):
                lexicon[word][k] = (lexicon[word][k] + entry[k]) / 2
        else: lexicon[word] = entry
    print(f"  NRC VAD [{lang}]: {len(lexicon):,} entries")
    return lexicon


def load_nrc_vad_bilingual() -> dict:
    """Merge the Catalan and Spanish NRC-VAD lexicons for bilingual patients.
    Both derive from the same English source, so labels are consistent; merging
    improves coverage. Catalan overwrites Spanish on conflict, since most
    bilingual patients here are Catalan-dominant."""
    es = load_nrc_vad("es")
    ca = load_nrc_vad("ca")
    conflicts = sum(
        1 for w in ca if w in es
        and any(abs(ca[w][k] - es[w][k]) > 0.01 for k in ("valence", "arousal", "dominance"))
    )
    merged = {**es, **ca}   # Catalan overwrites Spanish on conflict
    print(f"  NRC VAD [ca+es merged]: {len(merged):,} entries "
          f"({conflicts} minor conflicts resolved in favour of Catalan)")
    return merged


def load_clinical(path: Path) -> pd.DataFrame:
    """Read the clinical metadata CSV, normalising column names and the
    caliber_code column to stripped lowercase for reliable merging."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
    print(f"  Clinical data: {len(df)} patients")
    return df


# ── Text processing ────────────────────────────────────────────────────────────
def read_text(filepath: Path) -> str:
    """Read a transcript file trying several encodings, strip bracketed
    anonymisation placeholders (e.g. "[name]"), and return the cleaned text
    (empty string if the file cannot be decoded)."""
    for enc in ("utf-8", "utf-8-sig", "windows-1252", "latin-1"):
        try: return _ANON_RE.sub(" ", filepath.read_text(encoding=enc)).strip()
        except UnicodeDecodeError: continue
    print(f"  ⚠ Cannot decode {filepath.name}"); return ""


def preprocess_from_doc(doc, remove_stopwords: bool, keep_pronouns: bool = True) -> tuple:
    """Turn a spaCy doc into parallel lists of surface forms and lemmas,
    keeping only alphabetic tokens. Optionally drops stopwords, but retains a
    curated multilingual set of pronouns because they carry clinical
    information. Returns (surface_tokens, lemma_tokens)."""
    _PRONOUNS = {
        "i", "me", "my", "myself", "we", "our", "you",
        "he", "she", "it", "they", "them", "his", "her", "its", "their",
        "yo", "mi", "nosotros", "tú", "él", "ella", "ellos", "su", "sus",
        "jo", "em", "nosaltres", "tu", "ell", "ella", "ells", "elles",
    }
    surface, lemmas = [], []
    for token in doc:
        if token.is_space or token.is_punct or not token.is_alpha: continue
        s = token.lower_
        l = (token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.lower_).strip()
        if not s or not l: continue
        if remove_stopwords and token.is_stop:
            if not (keep_pronouns and l in _PRONOUNS): continue
        surface.append(s); lemmas.append(l)
    return surface, lemmas


# ── Network construction ───────────────────────────────────────────────────────
def build_cooccurrence_network(surface_tokens: list, window: int) -> nx.Graph:
    """Build an undirected co-occurrence network in which nodes are unique
    surface tokens and edges connect tokens appearing within `window` positions
    of each other. Edge weights use a Jaccard-style normalisation
    (raw_count / (freq_u + freq_v - raw_count))."""
    G    = nx.Graph()
    G.add_nodes_from(set(surface_tokens))
    freq = Counter(surface_tokens)
    n    = len(surface_tokens)
    for i, t1 in enumerate(surface_tokens):
        for j in range(i + 1, min(i + window + 1, n)):
            t2 = surface_tokens[j]
            if t1 == t2: continue
            if G.has_edge(t1, t2): G[t1][t2]["raw_count"] += 1
            else:                  G.add_edge(t1, t2, raw_count=1)
    for u, v, data in G.edges(data=True):
        raw   = data["raw_count"]
        denom = freq[u] + freq[v] - raw
        G[u][v]["weight"] = (raw / denom) if denom else 0.0
    return G


def annotate_vad(G: nx.Graph, surface_tokens: list,
                 lemma_tokens: list, vad_lex: dict) -> nx.Graph:
    """Annotate each network node with valence/arousal/dominance scores and
    categorical labels (positive/negative/neutral, high/low) from the VAD
    lexicon, matching first on the surface form and then on the lemma. Nodes
    with no lexicon entry receive NaN scores and neutral labels."""
    surf_to_lemma = dict(zip(surface_tokens, lemma_tokens))
    for node in G.nodes():
        entry = vad_lex.get(node) or vad_lex.get(surf_to_lemma.get(node, ""))
        if entry:
            v, a, d = entry["valence"], entry["arousal"], entry["dominance"]
            G.nodes[node].update({
                "valence": v, "arousal": a, "dominance": d,
                "valence_label":   ("positive" if v >= VAD_POS_THRESHOLD else
                                    "negative" if v <= VAD_NEG_THRESHOLD else "neutral"),
                "arousal_label":   ("high" if a >= VAD_POS_THRESHOLD else
                                    "low"  if a <= VAD_NEG_THRESHOLD else "neutral"),
                "dominance_label": ("high" if d >= VAD_POS_THRESHOLD else
                                    "low"  if d <= VAD_NEG_THRESHOLD else "neutral"),
            })
        else:
            G.nodes[node].update({
                "valence": np.nan, "arousal": np.nan, "dominance": np.nan,
                "valence_label": "neutral",
                "arousal_label": "neutral",
                "dominance_label": "neutral",
            })
    return G


# ── Triangle helpers ───────────────────────────────────────────────────────────
def _list_triangles(G: nx.Graph) -> list:
    """All (u, v, w) triangles, each yielded once (u < v < w lexicographically).
    Uses set intersection — faster than repeated has_edge calls."""
    adj  = {n: set(G.neighbors(n)) for n in G.nodes()}
    tris = []
    for u in G.nodes():
        for v in adj[u]:
            if v <= u: continue
            for w in adj[u] & adj[v]:
                if w > v:
                    tris.append((u, v, w))
    return tris


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_topology_features(G: nx.Graph) -> dict:
    """Compute the standard set of graph-topology descriptors for one network:
    size, density, degree/clustering/centrality distributions, connectivity,
    path-length and efficiency measures, and modularity. Metrics that cannot be
    computed on a given graph are returned as NaN rather than raising."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    f    = {"n_nodes": n, "n_edges": m, "density": nx.density(G)}
    _nan_keys = [k for k in TOPOLOGY_BASE if k not in f]
    if n == 0:
        f.update({k: np.nan for k in _nan_keys}); return f

    deg = np.array([d for _, d in G.degree()])
    f.update({
        "mean_degree": float(np.mean(deg)), "std_degree": float(np.std(deg)),
        "skew_degree": float(_skew(deg)) if len(deg) > 2 else np.nan,
        "max_degree":  int(np.max(deg)),
    })
    try:    f["assortativity"] = nx.degree_assortativity_coefficient(G)
    except: f["assortativity"] = np.nan

    clust = np.array(list(nx.clustering(G).values()))
    f.update({
        "mean_clustering": float(np.mean(clust)), "std_clustering": float(np.std(clust)),
        "skew_clustering": float(_skew(clust)) if len(clust) > 2 else np.nan,
    })

    try:
        pr = np.array(list(nx.pagerank(G, weight="weight").values()))
        f.update({"mean_pagerank": float(np.mean(pr)), "std_pagerank": float(np.std(pr)),
                  "skew_pagerank": float(_skew(pr)) if len(pr) > 2 else np.nan})
    except: f.update({"mean_pagerank": np.nan, "std_pagerank": np.nan, "skew_pagerank": np.nan})

    betw = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    f.update({"mean_betweenness": float(np.mean(betw)), "std_betweenness": float(np.std(betw)),
               "max_betweenness": float(np.max(betw))})

    try:
        eig = np.array(list(nx.eigenvector_centrality(G, max_iter=500, weight="weight").values()))
        f.update({"mean_eigenvector": float(np.mean(eig)), "std_eigenvector": float(np.std(eig))})
    except: f.update({"mean_eigenvector": np.nan, "std_eigenvector": np.nan})

    close = np.array(list(nx.closeness_centrality(G).values()))
    f.update({"mean_closeness": float(np.mean(close)), "std_closeness": float(np.std(close))})

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    f.update({"n_components": len(comps), "largest_component_size": len(comps[0])})
    lcc = G.subgraph(comps[0]).copy()
    if len(lcc) > 1:
        try:    f.update({"diameter": nx.diameter(lcc),
                          "mean_shortest_path": nx.average_shortest_path_length(lcc)})
        except: f.update({"diameter": np.nan, "mean_shortest_path": np.nan})
    else:
        f.update({"diameter": 0, "mean_shortest_path": 0})

    f["global_efficiency"] = nx.global_efficiency(G)
    f["local_efficiency"]  = nx.local_efficiency(G)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms       = greedy_modularity_communities(G, weight="weight")
        f["modularity"] = nx.community.modularity(G, comms, weight="weight")
    except: f["modularity"] = np.nan
    return f


# ── Emotional homophily (Stella et al. 2019) ──────────────────────────────────
def compute_homophily(G: nx.Graph, attr: str,
                      n_null: int = N_HOMOPHILY_NULL) -> dict:
    """Emotional homophily (Stella et al. 2019): Kendall's tau between the VAD
    label encodings (positive=1, neutral=0, negative=-1) of connected node pairs,
    z-scored against 1,000 label-shuffling permutations that preserve label
    frequency (an efficient approximation of Stella's configuration-model null)."""
    from scipy.stats import kendalltau

    label_key = f"{attr}_label"
    prefix    = f"{attr}_"
    _l2n      = {"positive": 1, "high": 1, "neutral": 0,
                 "negative": -1, "low": -1}

    out = {
        f"{prefix}homophily":   np.nan,
        f"{prefix}homophily_z": np.nan,
    }

    edges = list(G.edges())
    if len(edges) < 5:
        return out

    nodes  = list(G.nodes())
    labels = {n: G.nodes[n].get(label_key, "neutral") for n in nodes}

    # Observed Kendall tau between label encodings of connected pairs
    ev = [(_l2n.get(labels.get(u, "neutral"), 0),
           _l2n.get(labels.get(v, "neutral"), 0))
          for u, v in edges]
    tau_obs, _ = kendalltau([x[0] for x in ev], [x[1] for x in ev])
    out[f"{prefix}homophily"] = float(tau_obs)

    # Null model: shuffle labels, preserve empirical frequency
    rng       = np.random.default_rng(42)
    label_arr = np.array([_l2n.get(labels[n], 0) for n in nodes])
    null_taus = []

    for _ in range(n_null):
        perm        = rng.permutation(label_arr)
        perm_labels = {nodes[i]: perm[i] for i in range(len(nodes))}
        ev_null     = [(perm_labels.get(u, 0), perm_labels.get(v, 0))
                       for u, v in edges]
        tau_null, _ = kendalltau([x[0] for x in ev_null],
                                  [x[1] for x in ev_null])
        null_taus.append(tau_null)

    arr = np.array(null_taus)
    sd  = arr.std()
    out[f"{prefix}homophily_z"] = (
        float((tau_obs - arr.mean()) / sd) if sd > 0 else np.nan
    )
    return out


# ── Structural balance (Teixeira et al. 2021, fully vectorised) ───────────────
def compute_structural_balance_features(G: nx.Graph, attr: str,
                                         triangles: list,
                                         n_null: int = N_BALANCE_NULL) -> dict:
    """Heider structural-balance statistics for one VAD dimension (Teixeira et
    al. 2021). Edge sign is negative if either endpoint is negative, positive if
    either is positive (and neither negative), excluded if both are neutral.
    DoB = fraction of valid triads with s_uv * s_vw * s_uw > 0, z-scored against
    1,000 label-shuffling permutations."""
    prefix  = f"{attr}_"
    nan_out = {f"{prefix}dob": np.nan, f"{prefix}pct_neg_triads": np.nan,
               f"{prefix}dob_z": np.nan}
    if not triangles or G.number_of_nodes() < 3:
        return nan_out

    label_key  = f"{attr}_label"
    node_list  = list(G.nodes())
    node_to_i  = {n: i for i, n in enumerate(node_list)}
    node_signs = np.array(
        [_LABEL_INT.get(G.nodes[n].get(label_key, "neutral"), 0) for n in node_list],
        dtype=np.int8,
    )

    try:
        tri_u = np.array([node_to_i[u] for u, _v, _w in triangles], dtype=np.int32)
        tri_v = np.array([node_to_i[v] for _u, v, _w in triangles], dtype=np.int32)
        tri_w = np.array([node_to_i[w] for _u, _v, w in triangles], dtype=np.int32)
    except KeyError:
        return nan_out

    def _es(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorised edge-sign: -1 if either endpoint negative,
        +1 if either positive, 0 if both neutral."""
        return np.where((a < 0) | (b < 0), np.int8(-1),
               np.where((a > 0) | (b > 0), np.int8(1), np.int8(0)))

    def _balance(signs: np.ndarray):
        su, sv, sw    = signs[tri_u], signs[tri_v], signs[tri_w]
        s_uv, s_uw, s_vw = _es(su, sv), _es(su, sw), _es(sv, sw)
        valid    = (s_uv != 0) & (s_uw != 0) & (s_vw != 0)
        total    = int(valid.sum())
        if total == 0: return 0, 0, 0
        product  = (s_uv * s_uw * s_vw)[valid]
        balanced = int((product > 0).sum())
        neg_all  = int(((s_uv[valid] < 0) & (s_uw[valid] < 0) & (s_vw[valid] < 0)).sum())
        return balanced, neg_all, total

    balanced, neg_all, total = _balance(node_signs)
    if total == 0: return nan_out

    dob    = balanced / total
    result = {f"{prefix}dob": dob, f"{prefix}pct_neg_triads": neg_all / total,
              f"{prefix}dob_z": np.nan}

    # Null model: shuffle node labels, preserve frequency
    rng_bal   = np.random.default_rng(42)
    null_dobs = []
    perm      = node_signs.copy()
    for _ in range(n_null):
        rng_bal.shuffle(perm)
        b, _, t = _balance(perm)
        if t > 0: null_dobs.append(b / t)

    if len(null_dobs) >= 10:
        arr = np.array(null_dobs)
        sd  = arr.std()
        result[f"{prefix}dob_z"] = float((dob - arr.mean()) / sd) if sd > 0 else np.nan
    return result


# ── Emotional complexity (Stella et al. 2022) ─────────────────────────────────
def compute_emotional_complexity(G: nx.Graph, triangles: list,
                                  n_null: int = 500) -> dict:
    """Emotional complexity c = fraction of incoherent triads (Stella et al.
    2022). Node-based (distinct from edge-based DoB): a triad is incoherent if it
    holds >=1 positive AND >=1 negative node (neutral nodes ignored). z-scored
    against two 500-iteration nulls: shuffled (preserves valence frequency) and
    uniform (equal 1/3 label probability)."""
    out = {
        "emotional_complexity":          np.nan,
        "emotional_coherence":           np.nan,
        "emotional_complexity_z_shuffled": np.nan,
        "emotional_complexity_z_uniform":  np.nan,
    }

    if not triangles or G.number_of_nodes() < 3:
        return out

    val_labels = {n: G.nodes[n].get("valence_label", "neutral")
                  for n in G.nodes()}

    def _count_triads(labels: dict) -> tuple:
        n_coherent = 0
        n_incoherent = 0
        for u, v, w in triangles:
            has_pos = (labels.get(u) == "positive" or
                       labels.get(v) == "positive" or
                       labels.get(w) == "positive")
            has_neg = (labels.get(u) == "negative" or
                       labels.get(v) == "negative" or
                       labels.get(w) == "negative")
            if has_pos and has_neg:
                n_incoherent += 1
            else:
                n_coherent += 1
        return n_coherent, n_incoherent

    nc, ni = _count_triads(val_labels)
    total  = nc + ni
    if total == 0:
        return out

    c = ni / total
    out["emotional_complexity"] = float(c)
    out["emotional_coherence"]  = float(1 - c)

    rng        = np.random.default_rng(42)
    nodes      = list(G.nodes())
    n_nodes    = len(nodes)
    label_list = [val_labels[n] for n in nodes]
    base_labels = np.array(label_list)

    # Shuffled null: preserve frequency, reshuffle positions
    shuffled_c = []
    for _ in range(n_null):
        perm           = rng.permutation(base_labels)
        shuffled_labels = {nodes[i]: perm[i] for i in range(n_nodes)}
        nc_s, ni_s     = _count_triads(shuffled_labels)
        t_s = nc_s + ni_s
        if t_s > 0:
            shuffled_c.append(ni_s / t_s)

    # Uniform null: equal probability for each label
    uniform_c = []
    choices   = ["positive", "negative", "neutral"]
    for _ in range(n_null):
        uniform_labels = {n: rng.choice(choices) for n in nodes}
        nc_u, ni_u     = _count_triads(uniform_labels)
        t_u = nc_u + ni_u
        if t_u > 0:
            uniform_c.append(ni_u / t_u)

    if len(shuffled_c) >= 10:
        arr = np.array(shuffled_c)
        sd  = arr.std()
        out["emotional_complexity_z_shuffled"] = (
            float((c - arr.mean()) / sd) if sd > 0 else np.nan
        )

    if len(uniform_c) >= 10:
        arr = np.array(uniform_c)
        sd  = arr.std()
        out["emotional_complexity_z_uniform"] = (
            float((c - arr.mean()) / sd) if sd > 0 else np.nan
        )

    return out


# ── Coverage diagnostics ───────────────────────────────────────────────────────
def compute_coverage_diagnostics(G: nx.Graph, triangles: list) -> dict:
    """Coverage diagnostics contextualising emotional metrics on short clinical
    transcripts: VAD coverage, fraction of non-neutral nodes, raw positive/
    negative/neutral counts, number of triangles, and number of valid DoB triads
    (all three edge-signs non-neutral). Transcripts with pct_nonneutral_nodes <
    0.10 or n_triangles == 0 are unreliable for triangle-dependent metrics."""
    nd      = dict(G.nodes(data=True))
    n_total = G.number_of_nodes()

    if n_total == 0:
        return {
            "n_nodes_total":       0,
            "n_nodes_vad_covered": 0,
            "pct_vad_covered":     np.nan,
            "n_positive_nodes":    0,
            "n_negative_nodes":    0,
            "n_neutral_nodes":     0,
            "pct_nonneutral_nodes": np.nan,
            "n_triangles":         0,
            "n_valid_triads_dob":  0,
        }

    val_labels = {n: d.get("valence_label", "neutral") for n, d in nd.items()}

    # VAD coverage: nodes with a real (non-NaN) VAD entry
    n_covered    = sum(1 for n, d in nd.items()
                       if not np.isnan(d.get("valence", float("nan"))))
    n_pos        = sum(1 for l in val_labels.values() if l == "positive")
    n_neg        = sum(1 for l in val_labels.values() if l == "negative")
    n_neu        = sum(1 for l in val_labels.values() if l == "neutral")
    n_nonneutral = n_pos + n_neg

    # Valid triads for DoB (all three edge-signs must be non-neutral)
    _l2n = {"negative": -1, "neutral": 0, "positive": 1}
    signs = {n: _l2n.get(val_labels[n], 0) for n in G.nodes()}
    n_valid_dob = 0
    for u, v, w in triangles:
        su, sv, sw = signs[u], signs[v], signs[w]
        s_uv = -1 if (su < 0 or sv < 0) else (1 if (su > 0 or sv > 0) else 0)
        s_uw = -1 if (su < 0 or sw < 0) else (1 if (su > 0 or sw > 0) else 0)
        s_vw = -1 if (sv < 0 or sw < 0) else (1 if (sv > 0 or sw > 0) else 0)
        if s_uv != 0 and s_uw != 0 and s_vw != 0:
            n_valid_dob += 1

    return {
        "n_nodes_total":        n_total,
        "n_nodes_vad_covered":  n_covered,
        "pct_vad_covered":      n_covered / n_total,
        "n_positive_nodes":     n_pos,
        "n_negative_nodes":     n_neg,
        "n_neutral_nodes":      n_neu,
        "pct_nonneutral_nodes": n_nonneutral / n_total,
        "n_triangles":          len(triangles),
        "n_valid_triads_dob":   n_valid_dob,
    }


# ── VAD network features ───────────────────────────────────────────────────────
def compute_vad_network_features(G: nx.Graph, triangles: list) -> dict:
    """Network-specific VAD features: composition, homophily, structural balance,
    emotional complexity and coverage diagnostics. Aggregate statistics such as
    mean_valence are computed in features/psycholinguistics.py."""
    out     = {}
    nd      = dict(G.nodes(data=True))
    n_total = G.number_of_nodes()

    val_labels = {n: d.get("valence_label",   "neutral") for n, d in nd.items()}
    ar_labels  = {n: d.get("arousal_label",   "neutral") for n, d in nd.items()}
    dom_labels = {n: d.get("dominance_label", "neutral") for n, d in nd.items()}

    # ── Network composition ────────────────────────────────────────────────────
    vl = list(val_labels.values())
    out.update({
        "pct_positive_nodes": vl.count("positive") / n_total if n_total else np.nan,
        "pct_negative_nodes": vl.count("negative") / n_total if n_total else np.nan,
        "pct_neutral_nodes":  vl.count("neutral")  / n_total if n_total else np.nan,
    })

    ar = list(ar_labels.values())
    out.update({
        "pct_high_arousal_nodes": ar.count("high") / n_total if n_total else np.nan,
        "pct_low_arousal_nodes":  ar.count("low")  / n_total if n_total else np.nan,
    })

    dm = list(dom_labels.values())
    out.update({
        "pct_high_dominance_nodes": dm.count("high") / n_total if n_total else np.nan,
        "pct_low_dominance_nodes":  dm.count("low")  / n_total if n_total else np.nan,
    })

    # ── Emotional homophily (Stella et al. 2019) ───────────────────────────────
    for attr in ("valence", "arousal", "dominance"):
        out.update(compute_homophily(G, attr))

    # ── LCC valence profile ────────────────────────────────────────────────────
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if comps:
        lcc_nodes = comps[0]
        lcc_lbls  = [val_labels[n] for n in lcc_nodes]
        out["pct_pos_in_lcc"] = (lcc_lbls.count("positive") / len(lcc_lbls)
                                  if lcc_lbls else np.nan)
    else:
        out["pct_pos_in_lcc"] = np.nan

    # ── Structural balance (Teixeira et al. 2021) ─────────────────────────────
    for attr in ("valence", "arousal", "dominance"):
        out.update(compute_structural_balance_features(G, attr, triangles))

    # ── Emotional complexity (Stella et al. 2022; valence only) ───────────────
    out.update(compute_emotional_complexity(G, triangles))

    # ── Coverage diagnostics ──────────────────────────────────────────────────
    out.update(compute_coverage_diagnostics(G, triangles))

    return out


# ── Clinical merge ─────────────────────────────────────────────────────────────
def _parse_caliber_code(filename: str) -> str:
    """Extract the patient caliber_code from a transcript filename by stripping
    the known task suffix (e.g. "_emotion", "_bdae")."""
    stem = Path(filename).stem
    for suffix in ("_emotion", "_clinical_evaluation", "_bdae"):
        if stem.endswith(suffix): return stem[: -len(suffix)].strip().lower()
    return stem.strip().lower()


def _merge_with_clinical(df: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the per-patient feature frame with the clinical metadata on
    caliber_code, reconciling any duplicated language column and folding the
    mixed-language state via fold_mixed. Returns the merged frame."""
    keep_cols = [c for c in
                 ["caliber_code", "symptom_state", "language", "ymrs", "hdrs", "sex", "age", "visit"]
                 if c in clinical.columns]
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
    merged = df.merge(clinical[keep_cols], on="caliber_code", how="inner",
                      suffixes=("_feat", "_clin"))
    for col in ("language",):
        feat, clin = f"{col}_feat", f"{col}_clin"
        if feat in merged and clin in merged:
            merged[col] = merged[clin].fillna(merged[feat])
            merged = merged.drop(columns=[feat, clin])
        elif feat in merged: merged = merged.rename(columns={feat: col})
        elif clin in merged: merged = merged.rename(columns={clin: col})
    return fold_mixed(merged)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_cooccurrence_nrc_analysis(longitudinal: bool = False):
    """Run the full co-occurrence emotion pipeline: load clinical data, spaCy
    models and VAD lexicons; build and annotate a network per (patient, task,
    window) experiment; extract topology and VAD-network features; merge with
    clinical metadata; and write patient-level, coverage and per-/cross-
    experiment CSVs. In longitudinal mode it uses the longitudinal clinical file
    and output directory and stops after the session-level CSV (statistics
    handled by s11)."""
    clinical_path = Path(AUDIO_DATA_LONGITUDINAL if longitudinal
                         else AUDIO_DATA_UNIQUE)
    output_dir    = RESULTS_DIR / ("longitudinal/cooccurrence_emotion" if longitudinal
                                    else "cooccurrence_emotion")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "LONGITUDINAL" if longitudinal else "CROSS-SECTIONAL"
    print("═" * 70)
    print(f"CO-OCCURRENCE NETWORK ANALYSIS — {mode}")
    print(f"  Monologue root : {MONOLOGUE_ROOT}")
    print(f"  Output dir     : {output_dir}")
    print("═" * 70)

    print("\n[1/4] Clinical data…")
    clinical = load_clinical(clinical_path)
    _lmap    = {"english": "en", "spanish": "es", "catalan": "ca", "mix": "es"}
    lang_map = {}
    if "language" in clinical.columns:
        for _, row in clinical.iterrows():
            lang_map[str(row["caliber_code"]).strip().lower()] = _lmap.get(
                str(row.get("language", "")).strip().lower(), "unknown")
    print(f"  Language map: {len(lang_map)} entries  "
          f"({sum(v=='unknown' for v in lang_map.values())} unknown)")

    print("\n[2/4] spaCy models…")
    nlp_models = load_spacy_models()

    print("\n[3/4] NRC VAD lexicons…")
    vad_all = {
        "en":    load_nrc_vad("en"),
        "es":    load_nrc_vad_bilingual(),   # merged ca+es for bilingual patients
        "ca":    load_nrc_vad_bilingual(),
    }

    # ── Build (task_dir, window, remove_sw) → [exp_name] index ───────────────
    td_win_exps: dict[tuple, list] = defaultdict(list)
    for exp in EXPERIMENTS:
        for td in exp["task_dirs"]:
            td_win_exps[(td, exp["window"], exp["remove_sw"])].append(exp["name"])

    td_rsw_windows: dict[tuple, set] = defaultdict(set)
    for exp in EXPERIMENTS:
        for td in exp["task_dirs"]:
            td_rsw_windows[(td, exp["remove_sw"])].add(exp["window"])

    exp_record_sets: dict[str, list] = defaultdict(list)

    # ── Feature-extraction loop ───────────────────────────────────────────────
    print("\n[4/4] Building feature records…")
    for lang in ("es", "ca", "en"):
        vad = vad_all.get(lang, {})
        nlp = nlp_models.get(lang)

        for (task_dir_name, remove_sw), windows in td_rsw_windows.items():
            task_dir = MONOLOGUE_ROOT / task_dir_name
            if not task_dir.exists(): continue

            for fp in sorted(task_dir.glob("*.txt")):
                code = _parse_caliber_code(fp.name)
                if lang_map.get(code) != lang: continue

                text = read_text(fp)
                if not text: continue
                doc = nlp(text.lower()) if nlp else None
                if doc is None: continue

                surface, lemmas = preprocess_from_doc(doc, remove_sw)
                if len(surface) < 3: continue

                for window in sorted(windows):
                    key = (task_dir_name, window, remove_sw)
                    if key not in td_win_exps: continue

                    G         = build_cooccurrence_network(surface, window)
                    G         = annotate_vad(G, surface, lemmas, vad)
                    triangles = _list_triangles(G)

                    record = {
                        "caliber_code": code, "task": task_dir_name,
                        "language": lang, "word_count": len(surface),
                    }
                    record.update(extract_topology_features(G))
                    record.update(compute_vad_network_features(G, triangles))

                    for exp_name in td_win_exps[key]:
                        exp_record_sets[exp_name].append(record)

        print(f"  [{lang}] done.")

    # ── Per-experiment analysis ────────────────────────────────────────────────
    print(f"\n{'═'*70}\n  PER-EXPERIMENT ANALYSIS\n{'═'*70}")
    all_exp_dfs:      dict[str, pd.DataFrame] = {}
    long_rows:        list                    = []
    all_patient_rows: list                    = []

    # Coverage diagnostic columns — excluded from statistical analysis
    _COVERAGE_COLS = {
        "n_nodes_total", "n_nodes_vad_covered", "pct_vad_covered",
        "n_positive_nodes", "n_negative_nodes", "n_neutral_nodes",
        "pct_nonneutral_nodes", "n_triangles", "n_valid_triads_dob",
    }

    for exp in EXPERIMENTS:
        name    = exp["name"]
        records = exp_record_sets.get(name, [])
        print(f"\n{'─'*62}\n  {name}  |  {exp['note']}\n{'─'*62}")
        if not records: print("  ⚠ No records — skipping."); continue

        df = pd.DataFrame(records)
        df = _merge_with_clinical(df, clinical)
        print(f"  {len(df)} patients after merge.")
        if len(df) < 10: print("  ⚠ Insufficient data — skipping."); continue

        all_feat_cols = [c for c in (TOPOLOGY_BASE + VAD_NETWORK_FEATURES)
                         if c in df.columns]
        covariate_cols = [c for c in ["language", "word_count", "ymrs", "hdrs", "sex", "age"]
                          if c in df.columns]
        pat_cols = ["caliber_code", "symptom_state"] + covariate_cols + all_feat_cols
        if "visit" in df.columns:
            pat_cols.append("visit")
        pat = df[pat_cols].copy()
        pat["task_set"]   = exp["task_set"]
        pat["window"]     = exp["window"]
        pat["experiment"] = exp["name"]
        all_patient_rows.append(pat)

        topo_cols   = [b for b in TOPOLOGY_BASE if b in df.columns]
        vad_cols    = [c for c in VAD_NETWORK_FEATURES if c in df.columns]
        metric_cols = [
            c for c in (topo_cols + vad_cols)
            if c in df.columns
            and df[c].nunique() > 1
            and c not in _VERBOSITY_PROXIES
            and c not in _COVERAGE_COLS      # exclude diagnostics from stats
        ]

        df_exp = analyze_experiment(df, metric_cols)
        print_experiment_summary(name, df_exp)
        all_exp_dfs[name] = df_exp

        df_tagged = df_exp.copy()
        df_tagged["experiment"] = name
        df_tagged["task_set"]   = exp["task_set"]
        df_tagged["window"]     = exp["window"]
        long_rows.append(df_tagged)

    # ── Patient-level output ──────────────────────────────────────────────────
    if all_patient_rows:
        pat_df = pd.concat(all_patient_rows, ignore_index=True)

        if longitudinal:
            out_pat = output_dir / "cooccurrence_emotion_sessions.csv"
            pat_df.to_csv(out_pat, index=False)
            print(f"\n  Sessions saved → {out_pat}")
            print(f"    {len(pat_df)} sessions, "
                  f"{pat_df['caliber_code'].nunique()} patients")
            print("  longitudinal=True — skipping cross-sectional analysis.")
            return all_exp_dfs, pd.DataFrame()

        out_pat = output_dir / "cooccurrence_emotion_patient_values.csv"
        pat_df.to_csv(out_pat, index=False)
        print(f"\n  Patient-level features → {out_pat}")
        print(f"    {len(pat_df)} observations, "
              f"{pat_df['caliber_code'].nunique()} patients, "
              f"{pat_df['experiment'].nunique()} experiments")

        # ── Coverage diagnostics summary ──────────────────────────────────────
        coverage_cols = [
            "caliber_code", "symptom_state", "task_set",
            "window", "experiment", "word_count",
            "n_nodes_total", "n_nodes_vad_covered", "pct_vad_covered",
            "n_positive_nodes", "n_negative_nodes", "n_neutral_nodes",
            "pct_nonneutral_nodes", "n_triangles", "n_valid_triads_dob",
        ]
        cov_cols_present = [c for c in coverage_cols if c in pat_df.columns]
        cov_df = pat_df[cov_cols_present].copy()

        out_cov = output_dir / "cooccurrence_emotion_coverage.csv"
        cov_df.to_csv(out_cov, index=False)
        print(f"  Coverage diagnostics   → {out_cov}")

        print("\n  Coverage summary (mean across patients, by symptom state):")
        diag_cols = ["pct_vad_covered", "pct_nonneutral_nodes",
                     "n_triangles", "n_valid_triads_dob"]
        diag_present = [c for c in diag_cols if c in cov_df.columns]
        if "symptom_state" in cov_df.columns and diag_present:
            summary = (cov_df.groupby("symptom_state")[diag_present]
                       .agg(["mean", "median", "min", "max"])
                       .round(3))
            print(summary.to_string())

        if "pct_nonneutral_nodes" in cov_df.columns:
            low_cov = cov_df[cov_df["pct_nonneutral_nodes"] < 0.10]
            if len(low_cov) > 0:
                print(f"\n  ⚠ {len(low_cov)} transcripts with <10% non-neutral nodes "
                      f"(DoB/complexity unreliable):")
                id_cols = [c for c in ["caliber_code", "experiment",
                                        "pct_nonneutral_nodes", "n_triangles"]
                           if c in low_cov.columns]
                print(low_cov[id_cols].to_string(index=False))

        if "n_triangles" in cov_df.columns:
            no_tri = cov_df[cov_df["n_triangles"] == 0]
            if len(no_tri) > 0:
                print(f"\n  ⚠ {len(no_tri)} transcripts with zero triangles "
                      f"(DoB/complexity = NaN).")

    # ── Cross-experiment consistency ──────────────────────────────────────────
    print(f"\n{'═'*70}\n  CROSS-EXPERIMENT CONSISTENCY\n{'═'*70}")
    if long_rows:
        long_df    = pd.concat(long_rows, ignore_index=True)
        summary_df = consistency_across_experiments(long_df)
        print_consistency_summary(summary_df)
        long_df.to_csv(output_dir / "cooccurrence_emotion_long.csv", index=False)
        summary_df.to_csv(output_dir / "cooccurrence_emotion_consistency.csv", index=False)
    else:
        print("  ⚠ No experiment results to aggregate.")
        summary_df = pd.DataFrame()

    return all_exp_dfs, summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Co-occurrence Emotion Network Analysis"
    )
    parser.add_argument(
        "--longitudinal",
        action="store_true",
        default=False,
        help="Extract features for longitudinal dataset. "
             "Statistical analysis handled by analysis/longitudinal/longitudinal_analysis.py.",
    )
    args = parser.parse_args()
    run_cooccurrence_nrc_analysis(longitudinal=args.longitudinal)