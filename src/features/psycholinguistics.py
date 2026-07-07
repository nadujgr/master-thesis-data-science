"""
Psycholinguistic (meaning-derived) features.

Computes meaning-derived features from patient speech transcripts and tests
whether they distinguish mood states. Using spaCy models plus the NRC emotion
and VAD lexicons, it extracts features across several families: emotion (NRC
z-scores and valence/arousal/dominance statistics), self-reference (language-fair
and pro-drop aware for Spanish and Catalan), discourse markers, part-of-speech
rates, named entities, and syntactic complexity. Features are merged with
clinical mood labels and passed through the shared per-task and cross-task
consistency analysis. Inputs are the monologue transcript directories, the
clinical CSV, the spaCy models and the NRC lexicon files; outputs are the
per-task, consistency, patient-feature and summary files under
results/psycholinguistic/ (or the longitudinal equivalent).
"""

# ── Standard library ──
import json
import logging
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

# ── Third-party ──
import numpy as np
import pandas as pd
import spacy

# ── Local ──
warnings.filterwarnings("ignore")

from src.stats_helpers import (
    analyze_experiment,
    consistency_across_experiments,
    fold_mixed,
    print_consistency_summary,
    print_experiment_summary,
)
from src.utils import json_safe as _json_safe
from src.utils import parse_code as _parse_code
from src.config import (
    AUDIO_DATA_UNIQUE,
    NRC_DIR,
    NRC_EMOLEX_FILES,
    NRC_VAD_FILES,
    RESULTS_DIR,
    SPACY_MODELS as CFG_SPACY_MODELS,
    TRANSCRIPT_BY_PART_MONOLOGUE,
    AUDIO_DATA_LONGITUDINAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_NAME = "psycholinguistic"
OUTPUT_DIR = Path(RESULTS_DIR) / LEVEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONOLOGUE_ROOT = Path(TRANSCRIPT_BY_PART_MONOLOGUE)
CLINICAL_PATH = Path(AUDIO_DATA_UNIQUE)
NRC_DIR_PATH = Path(NRC_DIR)
SPACY_PATHS = {k: Path(v) for k, v in CFG_SPACY_MODELS.items()}

# All available tasks
TASK_DIRS = {
    "emotion": "emotion_clean",
    "clinical_evaluation": "clinical_evaluation_clean",
    "bdae": "bdae_clean",
}

# Minimum word count for valid transcript
MIN_WORD_COUNT = 10

# Emotion categories
EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

# Null model parameters
N_NULL_SAMPLES = 1000

# NRC EmoLex/VAD filenames come from config (NRC_EMOLEX_FILES, NRC_VAD_FILES),
# resolved against NRC_DIR at load time.

# Anonymization pattern
_ANON_RE = re.compile(r"\[[^\]]{1,30}\]")

# Discourse markers (language-specific)
_FILLERS = {
    "en": {"uh", "um", "er", "eh", "hmm"},
    "es": {"bueno", "pues", "este", "osea"},
    "ca": {"be", "bé", "doncs", "llavors"},
}

# First-person pronouns (for self-reference)
_FIRST_PERSON_PRONOUNS = {
    "en": {"i", "me", "my", "myself", "mine"},
    "es": {"yo", "me", "mi", "mí", "conmigo"},
    "ca": {"jo", "em", "me", "el meu", "la meva", "mi"},
}


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════


def build_language_map(df_unique: pd.DataFrame) -> dict:
    """Build mapping from caliber_code to language code."""
    if "caliber_code" not in df_unique.columns or "language" not in df_unique.columns:
        return {}
    
    norm = {"english": "en", "spanish": "es", "catalan": "ca", "mix": "es"}
    
    return (
        df_unique[["caliber_code", "language"]]
        .dropna()
        .assign(
            caliber_code=lambda x: x["caliber_code"].astype(str).str.strip().str.lower(),
            language=lambda x: x["language"].astype(str).str.strip().str.lower(),
        )
        .set_index("caliber_code")["language"]
        .map(norm)
        .to_dict()
    )


def load_spacy_models() -> dict:
    """Load spaCy models for each language."""
    log.info("Loading spaCy models...")
    nlp = {}
    fallbacks = {
        "en": "en_core_web_md",
        "es": "es_core_news_md",
        "ca": "ca_core_news_sm",
    }
    
    for lang, path in SPACY_PATHS.items():
        try:
            if Path(path).exists():
                nlp[lang] = spacy.load(str(path))
                log.info(f"  spaCy [{lang}] loaded ({Path(path).name})")
            else:
                nlp[lang] = spacy.load(fallbacks[lang])
                log.info(f"  spaCy [{lang}] fallback '{fallbacks[lang]}'")
        except OSError:
            log.warning(f"  spaCy [{lang}] not found - using blank")
            nlp[lang] = spacy.blank(lang)
    
    return nlp


def load_emolex(lang: str) -> dict:
    """Load NRC EmoLex for a given language.

    Handles two on-disk formats:
      - LONG  (English v0.92): `word\temotion\t0/1`, no header,
        one line per word-emotion pair.
      - WIDE  (Spanish/Catalan): header row with emotion-named columns;
        source English word first, translated word last.
    """
    path = NRC_DIR_PATH / NRC_EMOLEX_FILES.get(lang, "")
    if not path.exists():
        log.warning(f"  NRC EmoLex not found [{lang}]: {path}")
        return {}

    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Detect format from the first non-empty line.
    first = next((ln for ln in lines if ln.strip()), "")
    first_parts = first.split("\t")
    is_long = (
        len(first_parts) == 3
        and first_parts[1].strip().lower() in set(EMOTIONS) | {"positive", "negative"}
    )

    lex: dict = {}

    if is_long:
        # word \t emotion \t association
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            word = parts[0].strip().lower()
            emo = parts[1].strip().lower()
            if not word or emo not in EMOTIONS:
                continue
            try:
                val = int(parts[2])
            except ValueError:
                continue
            if word not in lex:
                lex[word] = {e: 0 for e in EMOTIONS}
            lex[word][emo] = max(lex[word][emo], val)
    else:
        # wide format with header
        header = first_parts
        emo_cols = {e: i for i, e in enumerate(header) if e.lower() in EMOTIONS}
        word_col = 0 if lang == "en" else len(header) - 1
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            word = parts[word_col].strip().lower()
            if not word:
                continue
            entry = {e: 0 for e in EMOTIONS}
            for e, ci in emo_cols.items():
                try:
                    entry[e] = int(parts[ci])
                except (ValueError, IndexError):
                    pass
            if word in lex:
                lex[word] = {e: max(lex[word][e], entry[e]) for e in EMOTIONS}
            else:
                lex[word] = entry

    log.info(f"  NRC EmoLex [{lang}]: {len(lex):,} entries")
    return lex

def load_emolex_bilingual() -> dict:
    """Merge Catalan and Spanish NRC EmoLex to increase coverage for
    Catalan-Spanish bilingual patients. Both are translated from the same
    English source, so emotion labels are consistent across languages."""
    es = load_emolex("es")
    ca = load_emolex("ca")
    # On conflict take max per category: labels are binary (0/1), so a word
    # marked in either language should be captured.
    merged = {**es}
    for word, ca_entry in ca.items():
        if word in merged:
            merged[word] = {e: max(merged[word].get(e, 0), ca_entry.get(e, 0))
                            for e in EMOTIONS}
        else:
            merged[word] = ca_entry
    log.info(f"  NRC EmoLex [ca+es merged]: {len(merged):,} entries")
    return merged


def load_nrc_vad(lang: str) -> dict:
    """Load the NRC valence-arousal-dominance (VAD) lexicon for one language as a
    word -> {valence, arousal, dominance} mapping. Handles the language-specific
    column layout (English word first, translated word last) and averages scores
    for repeated words. Returns {} if the file or V/A/D columns are missing."""
    path = NRC_DIR_PATH / NRC_VAD_FILES.get(lang, "")
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
    """Merge Catalan and Spanish NRC VAD lexicons to increase coverage for
    Catalan-Spanish bilingual patients. Both are translated from the same English
    source. Catalan takes precedence on conflict, as most bilingual patients in
    this sample are Catalan-dominant."""
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


def load_clinical() -> pd.DataFrame:
    """Load clinical data."""
    df = pd.read_csv(CLINICAL_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _read_text(fp: Path) -> str:
    """Read text file with anonymization removal."""
    return _ANON_RE.sub(" ", fp.read_text(encoding="utf-8").strip())


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — EMOTION
# ══════════════════════════════════════════════════════════════════════════════


def compute_emotion_features(doc, emolex: dict, lang: str, rng_seed: int = 42) -> dict:
    """Compute NRC emotion z-scores: observed emotion-word count vs a null model
    of random sampling from the lexicon."""
    rng = np.random.default_rng(rng_seed)

    lemmas = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    ]

    obs = {e: 0 for e in EMOTIONS}
    n_emo = 0

    for l in lemmas:
        entry = emolex.get(l)
        if entry:
            if any(entry.get(e, 0) for e in EMOTIONS):
                n_emo += 1
            for e in EMOTIONS:
                obs[e] += entry.get(e, 0)

    out = {
        "n_emotion_tokens": n_emo,
        "lexicon_coverage_emolex": n_emo / len(lemmas) if lemmas else np.nan,
    }

    if n_emo == 0 or not emolex:
        for e in EMOTIONS:
            out[f"z_{e}"] = np.nan
        return out
    
    lex_words = list(emolex.keys())
    null = {e: [] for e in EMOTIONS}

    for _ in range(N_NULL_SAMPLES):
        samp = rng.choice(lex_words, size=n_emo, replace=True)
        for e in EMOTIONS:
            null[e].append(sum(emolex[w].get(e, 0) for w in samp))
    
    for e in EMOTIONS:
        arr = np.array(null[e], dtype=float)
        std = arr.std()
        out[f"z_{e}"] = float((obs[e] - arr.mean()) / std) if std > 0 else np.nan
    
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — VAD
# ══════════════════════════════════════════════════════════════════════════════


def compute_vad_features(doc, vad_lex: dict) -> dict:
    """Compute VAD (valence, arousal, dominance) features."""
    lemmas = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    ]
    
    vals, arrs, doms = [], [], []
    pos_count = neg_count = 0

    for l in lemmas:
        entry = vad_lex.get(l)
        if entry:
            v = entry["valence"]
            vals.append(v)
            arrs.append(entry["arousal"])
            doms.append(entry["dominance"])
            
            if v >= 0.6:
                pos_count += 1
            elif v <= 0.4:
                neg_count += 1
    
    n_lemmas = len(lemmas)
    
    out = {
        "mean_valence": float(np.mean(vals)) if vals else np.nan,
        "std_valence": float(np.std(vals)) if vals else np.nan,
        "mean_arousal": float(np.mean(arrs)) if arrs else np.nan,
        "std_arousal": float(np.std(arrs)) if arrs else np.nan,
        "mean_dominance": float(np.mean(doms)) if doms else np.nan,
        "std_dominance": float(np.std(doms)) if doms else np.nan,
        "pct_positive_words": pos_count / n_lemmas if n_lemmas else np.nan,
        "pct_negative_words": neg_count / n_lemmas if n_lemmas else np.nan,
        "lexicon_coverage_vad": len(vals) / n_lemmas if n_lemmas else np.nan,
    }
    
    # Emotional entropy from valence-category transitions
    if len(vals) >= 2:
        labels = ["pos" if v >= 0.6 else "neg" if v <= 0.4 else "neu" for v in vals]
        trans = sum(1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1])
        p = trans / (len(labels) - 1)
        out["emotional_entropy"] = (
            sum(-q * math.log2(q) for q in (p, 1 - p) if q > 0)
            if p > 0 and p < 1
            else 0.0
        )
    else:
        out["emotional_entropy"] = np.nan
    
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — SELF-REFERENCE (PRO-DROP AWARE)
# ══════════════════════════════════════════════════════════════════════════════


def compute_self_reference(doc, lang: str) -> dict:
    """Language-fair self-reference rate: explicit pronouns for English, plus
    1st-person singular verb forms for pro-drop Spanish/Catalan.

    First-person self-reference is elevated in depression (Rude et al. 2004;
    Brockmeyer et al. 2015; Amorese et al. 2025) and reduced in mania.
    """
    pronouns = _FIRST_PERSON_PRONOUNS.get(lang, _FIRST_PERSON_PRONOUNS["es"])

    n_tokens = 0
    n_pronouns = 0
    n_fp_verbs = 0

    for tok in doc:
        if tok.is_space or tok.is_punct or not tok.is_alpha:
            continue
        n_tokens += 1

        if tok.lower_ in pronouns:
            n_pronouns += 1

        # 1st-person singular verb (pro-drop languages only)
        if tok.pos_ in ("VERB", "AUX") and lang in ("es", "ca"):
            morph = tok.morph
            if morph.get("Person") == ["1"] and morph.get("Number") == ["Sing"]:
                n_fp_verbs += 1
    
    if n_tokens == 0:
        return {
            "first_person_rate": np.nan,
            "pronoun_rate": np.nan,
            "first_person_verb_rate": np.nan,
        }
    
    return {
        "first_person_rate": (n_pronouns + n_fp_verbs) / n_tokens,
        "pronoun_rate": n_pronouns / n_tokens,
        "first_person_verb_rate": n_fp_verbs / n_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — DISCOURSE
# ══════════════════════════════════════════════════════════════════════════════


def compute_discourse_features(doc, text: str, lang: str) -> dict:
    """Discourse markers: filler/hesitation rate and question rate."""
    fillers = _FILLERS.get(lang, _FILLERS["es"])

    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    filler_count = sum(1 for t in tokens if t.lower_ in fillers)

    n_sents = len(list(doc.sents))
    question_count = text.count("?")
    
    return {
        "filler_rate": filler_count / n_tokens if n_tokens > 0 else np.nan,
        "question_rate": question_count / n_sents if n_sents > 0 else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — POS
# ══════════════════════════════════════════════════════════════════════════════


def compute_pos_features(doc) -> dict:
    """Part-of-speech distribution."""
    pos_counts = Counter(tok.pos_ for tok in doc if not tok.is_space)
    n_tokens = sum(pos_counts.values())
    
    if n_tokens == 0:
        return {
            k: np.nan
            for k in [
                "noun_rate",
                "verb_rate",
                "adj_rate",
                "adv_rate",
                "content_word_rate",
                "function_word_rate",
            ]
        }
    
    return {
        "noun_rate": pos_counts.get("NOUN", 0) / n_tokens,
        "verb_rate": pos_counts.get("VERB", 0) / n_tokens,
        "adj_rate": pos_counts.get("ADJ", 0) / n_tokens,
        "adv_rate": pos_counts.get("ADV", 0) / n_tokens,
        "content_word_rate": (
            pos_counts.get("NOUN", 0) + pos_counts.get("VERB", 0) + pos_counts.get("ADJ", 0)
        )
        / n_tokens,
        "function_word_rate": (
            pos_counts.get("DET", 0) + pos_counts.get("CONJ", 0) + pos_counts.get("ADP", 0)
        )
        / n_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — ENTITIES
# ══════════════════════════════════════════════════════════════════════════════


def compute_entity_features(doc) -> dict:
    """Named entity mentions."""
    ents = list(doc.ents)
    n_tokens = len([t for t in doc if not t.is_space])
    
    if n_tokens == 0:
        return {"person_mention_rate": np.nan, "entity_rate": np.nan}
    
    person_count = sum(1 for e in ents if e.label_ in ("PER", "PERSON"))
    
    return {
        "person_mention_rate": person_count / n_tokens,
        "entity_rate": len(ents) / n_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — COMPLEXITY
# ══════════════════════════════════════════════════════════════════════════════


def compute_complexity_features(doc) -> dict:
    """Syntactic complexity."""
    n_sents = len(list(doc.sents))

    if n_sents == 0:
        return {"mean_dependency_distance": np.nan, "subordination_rate": np.nan}

    dists = []
    for tok in doc:
        if tok.head != tok:
            dists.append(abs(tok.i - tok.head.i))

    n_subordinate = sum(
        1 for tok in doc if tok.dep_ in ("relcl", "advcl", "ccomp", "xcomp")
    )
    
    return {
        "mean_dependency_distance": float(np.mean(dists)) if dists else np.nan,
        "subordination_rate": n_subordinate / n_sents,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def build_task_features(
    task_name: str,
    task_dir_name: str,
    lang_map: dict,
    nlp_models: dict,
    emolex_all: dict,
    vad_all: dict,
    clinical: pd.DataFrame,
    longitudinal: bool = False,
) -> pd.DataFrame | None:
    """Build feature dataframe for one task."""
    
    task_dir = MONOLOGUE_ROOT / task_dir_name
    if not task_dir.exists():
        log.warning(f"  Task directory not found: {task_dir}")
        return None
    
    records = []
    skipped_short = 0
    skipped_lang = 0

    all_codes = [_parse_code(fp.name) for fp in sorted(task_dir.glob("*.txt"))]
    in_map = [c for c in all_codes if c in lang_map]
    not_in_map = [c for c in all_codes if c not in lang_map]
    log.info(f"  Files: {len(all_codes)} total, {len(in_map)} in lang_map, "
            f"{len(not_in_map)} NOT in lang_map")
    if not_in_map:
        log.warning(f"  Missing from lang_map: {not_in_map[:5]}")

    for fp in sorted(task_dir.glob("*.txt")):
        code = _parse_code(fp.name)
        lang = lang_map.get(code, "unknown")
        
        if lang not in ("en", "es", "ca"):
            skipped_lang += 1
            continue
        
        nlp = nlp_models.get(lang)
        if nlp is None:
            skipped_lang += 1
            continue
        
        text = _read_text(fp)
        if not text or len(text.split()) < MIN_WORD_COUNT:
            skipped_short += 1
            continue

        try:
            doc = nlp(text)
        except Exception as e:
            log.warning(f"  spaCy processing failed for {code}: {e}")
            continue

        emolex = emolex_all.get(lang, {})
        vad = vad_all.get(lang, {})

        record = {
            "caliber_code": code,
            "task": task_name,
            "language": lang,
            "word_count": len(text.split()),
        }
        
        record.update(compute_emotion_features(doc, emolex, lang))
        record.update(compute_vad_features(doc, vad))
        record.update(compute_self_reference(doc, lang))
        record.update(compute_discourse_features(doc, text, lang))
        record.update(compute_pos_features(doc))
        record.update(compute_entity_features(doc))
        record.update(compute_complexity_features(doc))
        
        records.append(record)
    
    if not records:
        log.warning(f"  No valid transcripts found for {task_name}")
        return None
    
    if skipped_short:
        log.info(f"  Skipped {skipped_short} transcripts (too short)")
    if skipped_lang:
        log.info(f"  Skipped {skipped_lang} transcripts (unknown/missing language)")
    
    df = pd.DataFrame(records)
    df["caliber_code"] = df["caliber_code"].astype(str).str.strip().str.lower()

    clinical_cols = [
        c
        for c in [
            "caliber_code",
            "symptom_state",
            "ymrs",
            "hdrs",
            "language",
            "sex",
            "age",
            "visit"
        ]
        if c in clinical.columns
    ]
    
    n_before = len(df)
    merged = df.merge(
        clinical[clinical_cols], on="caliber_code", how="inner", suffixes=("_feat", "_clin")
    )
    
    if "language_feat" in merged.columns and "language_clin" in merged.columns:
        merged["language"] = merged["language_clin"].fillna(merged["language_feat"])
        merged = merged.drop(columns=["language_feat", "language_clin"])
    elif "language_feat" in merged.columns:
        merged = merged.rename(columns={"language_feat": "language"})
    elif "language_clin" in merged.columns:
        merged = merged.rename(columns={"language_clin": "language"})
    
    merged = fold_mixed(merged)
    n_after = len(merged)
    
    if n_after < n_before:
        log.info(f"  ⚠ Dropped {n_before - n_after} patients (code not in clinical data)")

    states = merged["symptom_state"].value_counts()
    log.info(f"  [{task_name}] {len(merged)} {'sessions' if longitudinal else 'patients'}")
    log.info(
        f"    States: "
        + " | ".join(f"{s}={states.get(s, 0)}" for s in ["mania", "depression", "euthymia"])
    )
    
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_psycholinguistic(
    tasks: list[str] | None = None,
    longitudinal: bool = False,
) -> None:
    """Run the psycholinguistic pipeline for the requested tasks: load spaCy
    models and NRC lexicons, extract all feature categories per task, save the
    patient (or session) feature table, and, in cross-sectional mode, run the
    per-task analysis and cross-task consistency summary. Longitudinal mode only
    saves session features and defers statistics to the longitudinal script."""

    tasks = tasks or list(TASK_DIRS.keys())

    clinical_path = Path(AUDIO_DATA_LONGITUDINAL if longitudinal else AUDIO_DATA_UNIQUE)
    output_dir    = Path(RESULTS_DIR) / ("longitudinal/psycholinguistic" if longitudinal
                                          else "psycholinguistic")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "LONGITUDINAL" if longitudinal else "CROSS-SECTIONAL"
    log.info("=" * 70)
    log.info(f"  LEVEL 2 — PSYCHOLINGUISTIC FEATURES ({mode})")
    log.info(f"  Tasks:  {', '.join(tasks)}")
    log.info(f"  Output: {output_dir}")
    log.info("=" * 70)

    log.info("\nLoading resources...")
    clinical = pd.read_csv(clinical_path)
    clinical.columns = clinical.columns.str.strip().str.lower()
    clinical["caliber_code"] = (
        clinical["caliber_code"].astype(str).str.strip().str.lower()
    )
    lang_map   = build_language_map(clinical)
    nlp_models = load_spacy_models()
    emolex_all = {
        "en": load_emolex("en"),
        "es": load_emolex_bilingual(),   # merged ca+es for bilingual patients
        "ca": load_emolex_bilingual(),
    }
    vad_all = {
        "en":    load_nrc_vad("en"),
        "es":    load_nrc_vad_bilingual(),   # merged ca+es for bilingual patients
        "ca":    load_nrc_vad_bilingual(),
    }

    log.info("\nExtracting features...")
    task_dfs = {}
    patient_rows = []
    
    for task_name in tasks:
        task_dir_name = TASK_DIRS.get(task_name)
        if not task_dir_name:
            log.warning(f"  Unknown task: {task_name}")
            continue
        
        log.info(f"\n[{task_name}]")
        df = build_task_features(
            task_name, task_dir_name, lang_map, nlp_models, emolex_all, vad_all, clinical, longitudinal=longitudinal
        )
        
        if df is not None and len(df) >= 10:
            task_dfs[task_name] = df
            patient_rows.append(df)
    
    if not task_dfs:
        log.error("\n✗ No tasks produced sufficient data.")
        return
    
    emotion_cols = [f"z_{e}" for e in EMOTIONS]

    vad_cols = [
        "mean_valence",
        "std_valence",
        "mean_arousal",
        "std_arousal",
        "mean_dominance",
        "std_dominance",
        "pct_positive_words",
        "pct_negative_words",
        "emotional_entropy",
    ]

    selfref_cols = ["first_person_rate", "pronoun_rate", "first_person_verb_rate"]

    discourse_cols = ["filler_rate", "question_rate"]

    pos_cols = [
        "noun_rate",
        "verb_rate",
        "adj_rate",
        "adv_rate",
        "content_word_rate",
        "function_word_rate",
    ]

    entity_cols = ["person_mention_rate", "entity_rate"]

    complex_cols = ["mean_dependency_distance", "subordination_rate"]

    log.info("\n" + "=" * 70)
    log.info("  PER-TASK ANALYSIS")
    log.info("=" * 70)
    
    all_results = []
    
    for task_name, df in task_dfs.items():
        log.info(f"\n{task_name.upper()}")
        log.info("─" * 60)

        all_metric_cols = (
            emotion_cols
            + vad_cols
            + selfref_cols
            + discourse_cols
            + pos_cols
            + entity_cols
            + complex_cols
        )
        metric_cols = [
            c for c in all_metric_cols if c in df.columns and df[c].nunique() > 1
        ]
        
        if len(df) < 10 or not metric_cols:
            log.warning(f"  Insufficient data for analysis")
            continue

        res = analyze_experiment(df, metric_cols)
        
        if not res.empty:
            res.insert(0, "experiment", task_name)
            res["task"] = task_name
            all_results.append(res)
            print_experiment_summary(task_name, res)
    
    if not all_results:
        log.error("\n✗ No experiments produced results.")
        return
    
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(OUTPUT_DIR / "psycholinguistic_per_task.csv", index=False)

    log.info("\n" + "=" * 70)
    log.info("  CROSS-TASK CONSISTENCY")
    log.info("=" * 70)
    
    consistency_df = consistency_across_experiments(results_df)
    print_consistency_summary(consistency_df)
    consistency_df.to_csv(OUTPUT_DIR / "psycholinguistic_consistency.csv", index=False)
    
    if patient_rows:
        patient_df = pd.concat(patient_rows, ignore_index=True)
        patient_df["experiment"] = patient_df["task"].astype(str)

        if longitudinal:
            out_csv = output_dir / "psycholinguistic_sessions.csv"
            patient_df.to_csv(out_csv, index=False)
            log.info(f"\n  Sessions saved: {out_csv}")
            log.info(f"    {len(patient_df)} sessions, "
                     f"{patient_df['caliber_code'].nunique()} patients")
        else:
            out_csv = output_dir / "psycholinguistic_patient_features.csv"
            patient_df.to_csv(out_csv, index=False)
            log.info(f"\n  Patient features saved: {out_csv}")

    if longitudinal:
        log.info("  longitudinal=True — features saved, skipping cross-sectional analysis.")
        log.info("  Run analysis/longitudinal/longitudinal_analysis.py for longitudinal statistics.")
        return

    log.info(f"\n{'═'*70}")
    log.info("  SUMMARY")
    log.info(f"{'═'*70}")

    n_tasks = len(task_dfs)
    n_robust_strict = int(consistency_df["robust_strict"].sum())
    n_robust_pole = int(consistency_df["robust_pole"].sum())

    log.info(f"  Tasks analyzed: {n_tasks}")
    log.info(f"  Features tested: {len(consistency_df)}")
    log.info(f"  Robust features (strict, exact order ≥70%): {n_robust_strict}")
    log.info(f"  Robust features (pole-based, pattern ≥60%): {n_robust_pole}")

    if n_robust_pole > 0:
        log.info(
            f"\n  ✓ {n_robust_pole} psycholinguistic feature(s) show cross-task robustness:"
        )
        robust_features = consistency_df[consistency_df["robust_pole"]]["metric"].tolist()
        for feat in robust_features:
            pattern = consistency_df[consistency_df["metric"] == feat]["most_common_pattern"].iloc[0]
            log.info(f"      • {feat} ({pattern})")
        log.info("    These capture emotional content, self-focus, and complexity.")
    else:
        log.info("\n  ⚠ No psycholinguistic features show cross-task robustness.")

    log.info(f"\n  All outputs saved to: {OUTPUT_DIR}")

    summary = {
        "n_tasks": n_tasks,
        "tasks": list(task_dfs.keys()),
        "n_patients_per_task": {task: len(df) for task, df in task_dfs.items()},
        "n_features": len(all_metric_cols),
        "n_robust_strict": int(n_robust_strict),
        "n_robust_pole": int(n_robust_pole),
        "robust_features_strict": consistency_df[
            consistency_df["robust_strict"]
        ]["metric"].tolist(),
        "robust_features_pole": consistency_df[
            consistency_df["robust_pole"]
        ]["metric"].tolist(),
    }
    
    json_path = OUTPUT_DIR / "psycholinguistic_results.json"
    with json_path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    
    log.info(f"  Summary JSON: {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 2 - Psycholinguistic Features (Task-Agnostic)"
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
        "--longitudinal",
        action="store_true",
        default=False,
        help="Extract features for longitudinal analysis (sessions instead of patients)."
    )

    args = parser.parse_args()
    tasks = args.tasks or list(TASK_DIRS.keys())

    run_psycholinguistic(tasks, longitudinal=args.longitudinal)