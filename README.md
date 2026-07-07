# Cognitive Network Analysis and Psycholinguistic Profiling of Bipolar Disorder States

This repository contains the analysis code for a master's thesis investigating how spoken
language changes across mood states in bipolar disorder. The study asks whether the
structure and content of patients' speech — recorded during clinical sessions in mania,
euthymia and depression — carries a measurable signature of mood state. The data are
clinical speech transcripts collected across multiple elicitation tasks (a picture
description task, an emotional narrative and a clinical interview) in Spanish, Catalan, and English.
The pipeline characterises each session along several complementary axes: speech-graph connectedness, semantic similarity; co-occurrence network topology, lexical and
psycholinguistic features, and topic structure within- and between-patients. These features are
then used for statistical comparison across mood states and for supervised classification.

## Repository structure

The `src/` directory is a Python package grouped by pipeline stage. Modules are imported
as `src.<stage>.<module>` (for example `src.features.lexical`), which is why the code is
run from the project root (see "Running the analysis").

```
master-thesis-final/
├── main.py                       # Orchestrates the full pipeline (uncomment stages to run)
├── requirements.txt              # Pinned Python dependencies
├── README.md
├── src/
│   ├── config.py                 # All paths and model/analysis settings (project-relative)
│   ├── utils.py                  # Shared I/O and filename-normalisation helpers
│   ├── stats_helpers.py          # Shared statistics (effect sizes, FDR, experiment summaries)
│   ├── preparation/
│   │   ├── data_preparation.py   # Cleaning, labelling, dedup, transcript matching, anonymisation
│   │   └── preprocessing.py      # Transcript segmentation into task parts / monologues
│   ├── description/
│   │   ├── metadata_analysis.py  # Between-patient metadata description
│   │   └── transcript_analysis.py# Transcript-level descriptive statistics
│   ├── features/
│   │   ├── topic_embeddings.py   # Sentence/document embedding generation
│   │   ├── lexical.py            # Lexical feature extraction and analysis
│   │   ├── psycholinguistics.py  # Psycholinguistic feature extraction and analysis
│   │   ├── speech_graphs.py      # Speech-graph construction and metrics
│   │   ├── cooccurrence_emotion.py       # Co-occurrence networks + NRC emotion lexicon
│   │   ├── semantic_similarity.py        # Semantic similarity
│   │   ├── semantic_similarity_network.py# Semantic similarity networks
│   │   └── topic.py              # Topic modelling
│   ├── analysis/
│   │   ├── between_patient_stats.py      # Between-patient statistics
│   │   ├── classification.py     # Supervised classification of mood state
│   │   ├── correlation_analysis.py       # Cross-feature correlation analysis
│   │   └── longitudinal/
│   │       ├── build_long_features.py    # Builds longitudinal feature tables
│   │       └── longitudinal_analysis.py  # Within-patient longitudinal analysis
│   └── diagnostics/
│       ├── check_language_switching.py       # Language-switching diagnostics
│       └── verify_patient_data_consistency.py# Cross-file consistency checks
├── data/      (not included — see "Data access")
├── results/   (not included)
├── graphs/    (not included)
└── models/    (not included — local NLP model weights)
```

```
⚠️  The data/, results/, and graphs/ directories are not included in this repository.
    They contain sensitive clinical data subject to privacy restrictions and are
    excluded from version control. See "Data access" below.
```

## Analysis pipeline

The stages below run in order; later stages consume the processed transcripts and feature
tables produced by earlier ones. Each stage lives in the correspondingly named script and
exposes a single entry-point function called from `main.py`.

1. **Data preparation** (`preparation/data_preparation.py`) — clean and standardise the
   raw transcript export, correct metadata labels, deduplicate to unique patients, match
   transcripts to patient records, and replace anonymisation placeholders.
2. **Preprocessing** (`preparation/preprocessing.py`) — segment each session by
   elicitation task and produce dialogue, monologue and embedding-ready text.
3. **Metadata and transcript description** (`description/metadata_analysis.py`,
   `description/transcript_analysis.py`) — describe the cohort and the transcript corpus.
4. **Embedding generation** (`features/topic_embeddings.py`) — encode monologues for the
   semantic and topic analyses.
5. **Feature extraction and per-feature analysis** (`features/`) — lexical (`lexical.py`),
   psycholinguistic (`psycholinguistics.py`), speech graphs (`speech_graphs.py`),
   co-occurrence + emotion (`cooccurrence_emotion.py`), semantic similarity
   (`semantic_similarity.py`), semantic similarity networks
   (`semantic_similarity_network.py`) and topic modelling (`topic.py`).
6. **Between-patient statistics** (`analysis/between_patient_stats.py`) — aggregate
   features across patients and compare mood states.
7. **Classification** (`analysis/classification.py`) — supervised prediction of mood
   state from the extracted features.
8. **Correlation analysis** (`analysis/correlation_analysis.py`) — relationships across
   feature families.
9. **Longitudinal analysis** (`analysis/longitudinal/`) — `build_long_features.py` builds
   per-patient longitudinal tables, `longitudinal_analysis.py` analyses within-patient
   change over time.

Shared statistical routines (effect sizes, FDR correction, experiment summaries) live in
`src/stats_helpers.py`; shared I/O helpers live in `src/utils.py`. One-off consistency
checks live in `src/diagnostics/`.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Before running anything, review `src/config.py`. All paths are defined relative to the
project root, so no absolute paths need editing, but you must create the `data/` tree and
place the clinical inputs where `config.py` expects them (see "Data access"). The model
directories referenced under `models/` (spaCy, fastText, XLM-RoBERTa) must also be present
for the embedding-based stages.

## Data access

The clinical transcript data cannot be shared publicly. They consist of anonymised speech
transcripts from bipolar disorder patients recorded during clinical sessions across
different mood states, together with the associated session metadata. Because these
recordings contain sensitive health information, they are subject to privacy and research-
ethics restrictions and are excluded from version control, along with all derived results
and figures. The code expects
the inputs described in `config.py`: a transcript export, a session metadata table, the
NRC emotion lexicons, and the local NLP models.

## Running the analysis

The whole pipeline is driven from `main.py`, run from the project root. Stages are called
sequentially; most are commented out so that individual stages can be run in isolation
without recomputing everything:

```bash
python main.py
```

To run a stage, uncomment its call in `main()` (and comment out the others). Because `src`
is a package that imports its own modules as `src.<stage>.<module>`, always run from the
project root — either through `main.py` or as a module, for example:

```bash
python -m src.features.speech_graphs
```

or from a script at the project root:

```python
from src.features.speech_graphs import run_speech_graph_analysis
run_speech_graph_analysis(metric_suffix="")   # raw metrics
```

Because later stages depend on the outputs of earlier ones, run them in the order given in
the pipeline section the first time through.

## Dependencies

The project targets Python 3.11+ and relies on the standard scientific Python stack
(numpy, pandas, scipy, scikit-learn, statsmodels), NLP libraries (spaCy, transformers,
sentence-transformers, fastText, NLTK, NRCLex, BERTopic), network analysis (networkx). All versions are pinned in
`requirements.txt`.

## Citation / thesis reference

```
N. Jager. "Cognitive Network Analysis and Psycholinguistic Profiling of Bipolar
Disorder States." Master's thesis, UPC, 2026.
```
