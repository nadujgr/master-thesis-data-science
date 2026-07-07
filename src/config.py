"""
Central configuration for the thesis pipeline.

Every path here is derived from the project root (the parent of this ``src``
directory), so the repository can be cloned or moved without editing absolute
paths. Before running any analysis, make sure the ``data/`` tree exists and
populate it with the clinical inputs described in the README — the sensitive
``data/``, ``results/`` and ``graphs/`` folders are not distributed with the
code. The two roots a new user is most likely to change are ``DATA_RAW`` /
``DATA_PROCESSED`` (where inputs live) and ``RESULTS_DIR`` / ``GRAPHS_DIR``
(where outputs are written).
"""

import os

# ── Project roots ──────────────────────────────────────────────────────────────
# BASE_DIR is the repository root (one level up from src/). All other paths hang
# off it, so nothing here is machine-specific.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")              # untouched source inputs
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")  # cleaned / derived data

DOCS_DIR = os.path.join(BASE_DIR, "docs")          # data dictionaries and reference docs
RESULTS_DIR = os.path.join(BASE_DIR, "results")    # statistical outputs (tables, CSVs)
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")      # figures and plots
MODELS_DIR = os.path.join(BASE_DIR, "models")      # local NLP model weights

# ── Metadata / audio tables ────────────────────────────────────────────────────
CALIBER_AUDIO_DIR = os.path.join(DATA_PROCESSED, "caliber_audio")

AUDIO_DATA = os.path.join(DATA_RAW, "caliber_audio.xlsx")
AUDIO_DATA_CLEAN = os.path.join(CALIBER_AUDIO_DIR, "caliber_audio_clean.csv")
AUDIO_DATA_LABELLED = os.path.join(CALIBER_AUDIO_DIR, "caliber_audio_labelled.csv")
AUDIO_DATA_UNIQUE = os.path.join(CALIBER_AUDIO_DIR, "caliber_audio_unique_patients.csv")
AUDIO_DATA_UNIQUE_MEANINGFUL = os.path.join(CALIBER_AUDIO_DIR, "caliber_audio_unique_meaningful.csv")
AUDIO_DATA_LONGITUDINAL = os.path.join(CALIBER_AUDIO_DIR, "caliber_audio_longitudinal.csv")

DICTIONARY = os.path.join(DOCS_DIR, "variable_dictionary_caliber_audio.xlsx")

# ── Transcripts ────────────────────────────────────────────────────────────────
# The manually checked transcript export is the working source. Earlier exports
# are kept here for provenance but are not read by the pipeline.
TRANSCRIPTS = os.path.join(DATA_RAW, "segments_llm_ready_anonymised checkedGA.xlsx")
TRANSCRIPTS_METADATA_MERGED = os.path.join(DATA_PROCESSED, "transcripts_metadata_merged.csv")

# Raw per-session transcript JSON exports and their cleaned counterparts
TRANSCRIPT_JSON_DIR = os.path.join(DATA_RAW, "transcript-identification")
TRANSCRIPT_JSON_DIR_CLEAN = os.path.join(DATA_PROCESSED, "transcript-identification")

# ── Transcript segmentation by task part ───────────────────────────────────────
# Sessions are split by elicitation task (BDAE picture description, emotion
# narrative, clinical evaluation) and stored both as dialogue and as concatenated
# monologue for downstream feature extraction.
TRANSCRIPT_BY_PART = os.path.join(DATA_PROCESSED, "transcript_by_part")

TRANSCRIPT_BY_PART_DIALOGUE = os.path.join(DATA_PROCESSED, "transcript_by_part_dialogue")
TRANSCRIPT_PART_BDAE_DIALOGUE = os.path.join(TRANSCRIPT_BY_PART_DIALOGUE, "bdae")
TRANSCRIPT_PART_EMOTION_DIALOGUE = os.path.join(TRANSCRIPT_BY_PART_DIALOGUE, "emotion")
TRANSCRIPT_PART_CLINICAL_DIALOGUE = os.path.join(TRANSCRIPT_BY_PART_DIALOGUE, "clinical_evaluation")

# Concatenated monologues (patient turns only) per task part
TRANSCRIPT_BY_PART_MONOLOGUE = os.path.join(DATA_PROCESSED, "transcript_by_part_monologue")
TRANSCRIPT_PART_BDAE_MONOLOGUE = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "bdae")
TRANSCRIPT_PART_EMOTION_MONOLOGUE = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "emotion")
TRANSCRIPT_PART_CLINICAL_MONOLOGUE = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "clinical_evaluation")

# Embedding-ready monologues (whitespace-normalised, ready for encoders)
TRANSCRIPT_PART_BDAE_EMBEDDING = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "bdae_embedding_ready")
TRANSCRIPT_PART_EMOTION_EMBEDDING = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "emotion_embedding_ready")
TRANSCRIPT_PART_CLINICAL_EMBEDDING = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "clinical_evaluation_embedding_ready")

# Cleaned monologues (anonymisation tokens and accent marks removed)
TRANSCRIPT_PART_BDAE_CLEAN = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "bdae_clean")
TRANSCRIPT_PART_EMOTION_CLEAN = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "emotion_clean")
TRANSCRIPT_PART_CLINICAL_CLEAN = os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "clinical_evaluation_clean")

# SpeechGraphs input folder
SPEECHGRAPH_DATA = os.path.join(DATA_PROCESSED, "speechgraph_data")

# ── Lexical resources ──────────────────────────────────────────────────────────
# NRC lexicons live under DATA_RAW/nrc. EmoLex supplies the eight NRC emotion
# categories; VAD supplies valence/arousal/dominance scores. Each analysis loads
# the file for the transcript language (English, Spanish or Catalan); filenames
# are resolved against NRC_DIR at load time.
NRC_DIR = os.path.join(DATA_RAW, "nrc")

# NRC EmoLex (emotion categories) — one file per language
NRC_EMOLEX_FILES = {
    "en": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "es": "Spanish-NRC-EmoLex.txt",
    "ca": "Catalan-NRC-EmoLex.txt",
}

# NRC-VAD (valence / arousal / dominance) — one file per language
NRC_VAD_FILES = {
    "en": "NRC-VAD-Lexicon.txt",
    "es": "Spanish-NRC-VAD-Lexicon.txt",
    "ca": "Catalan-NRC-VAD-Lexicon.txt",
}

# ── Models ─────────────────────────────────────────────────────────────────────
XLM_ROBERTA_DIR = os.path.join(MODELS_DIR, "xlm-roberta-base")

# Per-language fastText embedding models
FASTTEXT_LANG_MODELS = {
    "es": os.path.join(MODELS_DIR, "fasttext", "cc.es.300.bin"),
    "ca": os.path.join(MODELS_DIR, "fasttext", "cc.ca.300.bin"),
    "en": os.path.join(MODELS_DIR, "fasttext", "cc.en.300.bin"),
}
FASTTEXT_DEFAULT_LANG = "es"

# Per-language spaCy pipelines (transformer variants)
SPACY_MODELS = {
    "es": os.path.join(MODELS_DIR, "spacy", "es_dep_news_trf"),
    "ca": os.path.join(MODELS_DIR, "spacy", "ca_core_news_trf"),
    "en": os.path.join(MODELS_DIR, "spacy", "en_core_web_trf"),
}
SPACY_DEFAULT_LANG = "es"

# ── Results subdirectories ─────────────────────────────────────────────────────
DATA_OVERVIEW_RESULTS_DIR = os.path.join(RESULTS_DIR, "data_overview")
DIALOGUE_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "dialogue_analysis")
SPEECH_GRAPH_RESULTS_DIR = os.path.join(RESULTS_DIR, "speech_graph")
SEMANTIC_RESULTS_DIR = os.path.join(RESULTS_DIR, "semantic_similarity")
COOCCURRENCE_RESULTS_DIR = os.path.join(RESULTS_DIR, "cooccurrence")
CLASSIFICATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "classification")

# ── Graph (figure) subdirectories ──────────────────────────────────────────────
GRAPHS_OVERVIEW_DIR = os.path.join(GRAPHS_DIR, "overview")
GRAPHS_FINAL_OVERVIEW_DIR = os.path.join(GRAPHS_DIR, "final_overview")
GRAPHS_SPEECH_GRAPH_DIR = os.path.join(GRAPHS_DIR, "speech_graphs")
GRAPHS_SEMANTIC_DIR = os.path.join(GRAPHS_DIR, "semantic_similarity")
GRAPHS_CLASSIFICATION_DIR = os.path.join(GRAPHS_DIR, "classification")
