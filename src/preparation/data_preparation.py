"""
Consolidated data-preparation pipeline for the CALIBER Audio study. Runs five
stages in order: clean the raw audio metadata spreadsheet and standardise the
language field; apply reviewed episode-label corrections and derive the
score-based symptom state; select one representative visit per patient (the
unique-patient and "meaningful" cohorts); match diarised transcript JSONs to
patient records, split them by interview part, and produce dialogue and
patient-only monologue text; and replace anonymisation placeholders with
language-appropriate words for embedding input, plus accent-kept and
accent-stripped cleaned monologues. Inputs are the raw metadata spreadsheet and
transcript JSONs; outputs are the cleaned CSVs and per-part text exports.
"""
import os
import re
import json
import shutil
import unicodedata
from pathlib import Path
from itertools import groupby
from collections import Counter

import numpy as np
import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory

from src.config import (
    AUDIO_DATA,
    AUDIO_DATA_CLEAN,
    AUDIO_DATA_LABELLED,
    AUDIO_DATA_UNIQUE,
    AUDIO_DATA_UNIQUE_MEANINGFUL,
    TRANSCRIPT_JSON_DIR,
    TRANSCRIPT_JSON_DIR_CLEAN,
    TRANSCRIPT_BY_PART,
    TRANSCRIPT_BY_PART_DIALOGUE,
    TRANSCRIPT_PART_BDAE_DIALOGUE,
    TRANSCRIPT_PART_EMOTION_DIALOGUE,
    TRANSCRIPT_PART_CLINICAL_DIALOGUE,
    TRANSCRIPT_BY_PART_MONOLOGUE,
    TRANSCRIPT_PART_BDAE_MONOLOGUE,
    TRANSCRIPT_PART_EMOTION_MONOLOGUE,
    TRANSCRIPT_PART_CLINICAL_MONOLOGUE,
)

# Deterministic language detection: langdetect is stochastic unless seeded.
DetectorFactory.seed = 42

# Default language used whenever detection fails or metadata has no language.
FALLBACK_LANG = "spanish"

# Short ISO codes returned by langdetect mapped to the study's language names.
LANG_MAP = {"ca": "catalan", "es": "spanish", "en": "english"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataframe(path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def normalise_code_column(series):
    """Cast a caliber_code column to string, strip whitespace and lowercase it so
    filenames, JSON stems and metadata codes are comparable."""
    return series.astype(str).str.strip().str.lower()


def detect_language(text):
    """Guess a transcript's language as one of the study's language names.
    Anonymisation tags are stripped and very short texts fall back to the default,
    since langdetect is unreliable on tag noise and tiny fragments."""
    clean = re.sub(r"<[A-Z_]+>", "", text).strip()
    if len(clean) < 20:
        return FALLBACK_LANG
    try:
        return LANG_MAP.get(detect(clean), FALLBACK_LANG)
    except LangDetectException:
        return FALLBACK_LANG


def read_text_file(path):
    """Read a text file as UTF-8, falling back to windows-1252 for files saved
    with the Windows codepage."""
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="windows-1252", errors="replace")


def write_text_file(path, content):
    """Write text to a UTF-8 file, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Section 1: Initial data cleaning ──────────────────────────────────────────
# Keep the columns used downstream, fix confirmed data-entry errors, and fill the
# language field from transcripts (langdetect) where metadata is blank; remaining
# blanks default to Spanish and "mix" is folded into Catalan.

def run_data_cleaning():
    df_audio = pd.read_excel(AUDIO_DATA)

    # Columns retained for all downstream analysis; everything else is dropped.
    keep_cols = [
        "id", "visit", "caliber_code", "episode_type", "episode_time", "sex",
        "age", "ymrs", "hdrs", "panss_p", "panss_n", "panss_g", "language",
    ]
    cleaned_metadata = df_audio[keep_cols].copy()
    print("Columns remaining:", cleaned_metadata.shape[1])

    cleaned_metadata.loc[
        cleaned_metadata["caliber_code"] == "254_mde_bd_acute", "episode_type"
    ] = "depression"
    cleaned_metadata.loc[
        cleaned_metadata["caliber_code"] == "254_mde_bd_acute", "episode_time"
    ] = "acute"

    # Reviewed label corrections: caliber_code tokens cross-checked against
    # episode_type / episode_time.
    corrections = [
        ("44_eu_bd",            "euthymia",   "euthymia"),
        ("116_eu_bd",           "euthymia",   "euthymia"),
        ("177_eu_bd",           "euthymia",   "euthymia"),
        ("214_eu_bd",           "euthymia",   "euthymia"),
        ("136_eu_bd",           "euthymia",   "euthymia"),
        ("143_eu_bd",           "euthymia",   "euthymia"),
        ("151_eu_bd",           "euthymia",   "euthymia"),
        ("158_eu_bd",           "euthymia",   "euthymia"),
        ("205_eu_bd",           "euthymia",   "euthymia"),
        ("153_mde_bd_acute",    "depression", "acute"),
        ("190_mde_bd_acute",    "depression", "acute"),
        ("212_mde_bd_acute",    "depression", "acute"),
        ("223_mde_bd_acute",    "depression", "acute"),
        ("223_mde_bd_respremi", "depression", "remission"),
        ("159_mde_bd_acute",    "depression", "acute"),
        ("208_mde_bd_acute",    "depression", "acute"),
        ("260_mde_bd_acute",    "depression", "acute"),
        ("227_mde_bd_acute",    "depression", "acute"),
        ("68_mde_bd_acute",     "depression", "acute"),
        ("107_mde_bd_acute",    "depression", "acute"),
        ("68_me_acute",         "mania",      "acute"),
        ("107_me_acute",        "mania",      "acute"),
        ("254_mde_bd_acute",    "depression", "acute"),
        ("143_mde_bd_respremi", "depression", "remission"),
        ("201_mde_bd_respremi", "depression", "remission"),
        ("223_mde_bd_respremi", "depression", "remission"),
    ]
    for code, new_type, new_time in corrections:
        mask = cleaned_metadata["caliber_code"] == code
        cleaned_metadata.loc[mask, "episode_type"] = new_type
        cleaned_metadata.loc[mask, "episode_time"] = new_time
    print(f"Applied {len(corrections)} label corrections.")

    # Detect each participant's language from their monologues; scan the emotion
    # and BDAE folders, first occurrence of each code wins.
    task_dirs = {
        "emotion": TRANSCRIPT_PART_EMOTION_MONOLOGUE,
        "bdae":    TRANSCRIPT_PART_BDAE_MONOLOGUE,
    }
    detected_langs = {}
    for task, src_dir in task_dirs.items():
        for fpath in sorted(Path(src_dir).glob("*.txt")):
            code = (
                fpath.stem
                .replace(f"_{task}", "")
                .replace("_bdae", "")
                .replace("_emotion", "")
                .strip()
                .lower()
            )
            if code in detected_langs:
                continue
            detected_langs[code] = detect_language(read_text_file(fpath))

    print(f"  Language detected for {len(detected_langs)} participants")
    print(f"  Distribution: {Counter(detected_langs.values()).most_common()}")

    # Fill blank metadata languages from the detected values, then drop the helper.
    df_lang = pd.DataFrame(
        list(detected_langs.items()),
        columns=["caliber_code", "language_detected"],
    )
    cleaned_metadata = cleaned_metadata.merge(df_lang, on="caliber_code", how="left")
    if "language" in cleaned_metadata.columns:
        before = cleaned_metadata["language"].notna().sum()
        cleaned_metadata["language"] = cleaned_metadata["language"].fillna(
            cleaned_metadata["language_detected"]
        )
        after = cleaned_metadata["language"].notna().sum()
        cleaned_metadata = cleaned_metadata.drop(columns=["language_detected"])
        print(f"  Language column: {before} existing → {after} after fill")
    else:
        cleaned_metadata = cleaned_metadata.rename(
            columns={"language_detected": "language"}
        )

    # Any remaining blank defaults to Spanish; "mix" is treated as Catalan.
    cleaned_metadata.fillna({"language": FALLBACK_LANG}, inplace=True)
    cleaned_metadata["language"] = cleaned_metadata["language"].replace("mix", "catalan")
    print("  Final language distribution:")
    print(cleaned_metadata["language"].value_counts(dropna=False))

    cleaned_metadata.to_csv(AUDIO_DATA_CLEAN, index=False)
    return cleaned_metadata


# ── Section 2: Metadata labelling ─────────────────────────────────────────────
# Derive a score-based "symptom_state" from the YMRS and HDRS scales (≥ 8 on
# each), independent of episode_type, giving a reproducible severity label that
# later analyses can compare against the clinical label.

def run_metadata_labelling():
    labelled_metadata = load_dataframe(AUDIO_DATA_CLEAN)

    ymrs_full = labelled_metadata["ymrs"] >= 8
    hdrs_full = labelled_metadata["hdrs"] >= 8

    labelled_metadata["symptom_state"] = np.select(
        [
            ~hdrs_full & ~ymrs_full,   # both low     → euthymia
            hdrs_full  & ~ymrs_full,   # only HDRS ≥8 → depression
            ~hdrs_full & ymrs_full,    # only YMRS ≥8 → mania
            hdrs_full  & ymrs_full,    # both ≥8      → mixed
        ],
        ["euthymia", "depression", "mania", "mixed"],
        default="unknown",
    )
    print(labelled_metadata["symptom_state"].value_counts())

    labelled_metadata.to_csv(AUDIO_DATA_LABELLED, index=False)
    return labelled_metadata


# ── Section 3: Unique-patient selection ───────────────────────────────────────
# Reduce the visit-level table to one session per patient: single-visit patients
# kept as-is, multi-visit patients get a manually curated acute visit. A second
# "meaningful" cohort swaps in a later session for three patients whose acute
# session lacks an emotion transcript. Both cohorts are written to disk.

def run_unique_patient_selection():
    labelled_metadata = load_dataframe(AUDIO_DATA_LABELLED)

    # Patients contributing exactly one visit are always kept.
    single_visit_ids = (
        labelled_metadata.groupby("id").filter(lambda x: len(x) == 1)["id"]
    )

    # Manually chosen acute visit for each multi-visit patient.
    manually_selected = [
        "45_mde_bd_acute", "68_me_acute", "92_me_acute", "107_me_acute",
        "128_mde_bd_acute", "143_mde_bd_acute", "158_me_acute", "168_me_acute",
        "192_me_acute", "201_mde_bd_acute", "202_mde_bd_acute", "218_mde_bd_acute",
        "219_mde_bd_acute", "220_mde_bd_acute", "223_mde_bd_acute", "227_me_acute",
        "254_mde_bd_acute", "261_me_acute", "263_me_acute", "269_me_acute",
    ]
    unique_patients = labelled_metadata[
        labelled_metadata["id"].isin(single_visit_ids)
        | labelled_metadata["caliber_code"].isin(manually_selected)
    ].copy()
    unique_patients.to_csv(AUDIO_DATA_UNIQUE, index=False)

    print(f"Total rows: {len(unique_patients)}")
    print(f"  - From single-visit patients: "
          f"{labelled_metadata['id'].isin(single_visit_ids).sum()}")
    print(f"  - From manually selected visits: "
          f"{labelled_metadata['caliber_code'].isin(manually_selected).sum()}")
    print("=== Episode type breakdown ===")
    print(unique_patients["episode_type"].value_counts())
    print("=== Symptom state breakdown ===")
    print(unique_patients["symptom_state"].value_counts())

    # "Meaningful" cohort: swap in visits that actually have an emotion transcript
    # for the three patients whose acute session did not.
    manually_selected_meaningful = [
        "45_mde_bd_acute",   "68_me_acute",        "92_me_acute",
        "107_me_acute",      "128_mde_bd_acute",   "143_mde_bd_acute",
        "158_me_acute",
        "168_me_resp",       # was 168_me_acute (no emotion transcript)
        "192_me_acute",      "201_mde_bd_acute",
        "202_mde_bd_remi",   # was 202_mde_bd_acute (no emotion transcript)
        "218_mde_bd_acute",  "219_mde_bd_acute",   "220_mde_bd_acute",
        "223_mde_bd_acute",  "227_me_acute",       "254_mde_bd_acute",
        "261_me_remi",       # was 261_me_acute (no emotion transcript)
        "263_me_acute",
        "269_me_acute",
    ]
    unique_patients_meaningful = labelled_metadata[
        labelled_metadata["id"].isin(single_visit_ids)
        | labelled_metadata["caliber_code"].isin(manually_selected_meaningful)
    ].copy()
    unique_patients_meaningful.to_csv(AUDIO_DATA_UNIQUE_MEANINGFUL, index=False)
    return unique_patients, unique_patients_meaningful


# ── Section 4: Transcript identification ───────────────────────────────────────
# Match diarised transcript JSONs to metadata codes (applying manual filename
# fixes), copy the corrected JSONs to a clean folder, load them into one flat
# segment table, filter to the analysis visits, and re-emit per interview part as
# raw JSON, speaker-labelled dialogue text, and patient-only monologue text.

# Filename → correct caliber_code fixes for the transcript JSON files.
MANUAL_ID_FIXES = {
    "275_mania_bd_acute":  "275_me_acute",
    "107_me_resp":         "107_me_acute",
    "136_mde_bd_respremi": "136_eu_bd",
    "159_mde_bd_resp":     "159_mde_bd_acute",
    "205_mde_bd_respremi": "205_eu_bd",
    "208_eu_bd":           "208_mde_bd_acute",
    "264_mde_bd_resp":     "264_mde_bd_acute",
    "268_me_bd_acute":     "268_me_acute",
    "269_me_bd_acute":     "269_me_acute",
    "269_me_bd_resp":      "269_me_resp",
}

# The three interview parts kept for analysis.
PART_OUTPUT_DIRS = {
    "bdae":                os.path.join(TRANSCRIPT_BY_PART, "bdae"),
    "emotion":             os.path.join(TRANSCRIPT_BY_PART, "emotion"),
    "clinical_evaluation": os.path.join(TRANSCRIPT_BY_PART, "clinical_evaluation"),
}


def run_transcript_identification():
    df_audio_clean = load_dataframe(AUDIO_DATA_CLEAN)

    # Apply the manual filename fixes so JSON stems line up with caliber_code.
    json_names = sorted(
        f.replace(".json", "")
        for f in os.listdir(TRANSCRIPT_JSON_DIR)
        if f.endswith(".json")
    )
    json_names_fixed = [MANUAL_ID_FIXES.get(name, name) for name in json_names]

    duplicates = [n for n in json_names_fixed if json_names_fixed.count(n) > 1]
    if duplicates:
        print("Duplicates after fixing:", sorted(set(duplicates)))

    # Copy each (possibly renamed) JSON into the clean directory.
    os.makedirs(TRANSCRIPT_JSON_DIR_CLEAN, exist_ok=True)
    for orig_name, fixed_name in zip(json_names, json_names_fixed):
        src = os.path.join(TRANSCRIPT_JSON_DIR, orig_name + ".json")
        dst = os.path.join(TRANSCRIPT_JSON_DIR_CLEAN, fixed_name + ".json")
        shutil.copy2(src, dst)
    print(f"{len(json_names)} JSON files written to {TRANSCRIPT_JSON_DIR_CLEAN}")

    # Load every clean JSON into one flat segment table, tagging each segment with
    # its caliber_code and coercing the numeric columns.
    all_segments = []
    for fname in sorted(
        f for f in os.listdir(TRANSCRIPT_JSON_DIR_CLEAN) if f.endswith(".json")
    ):
        caliber_code = fname.replace(".json", "")
        fpath = os.path.join(TRANSCRIPT_JSON_DIR_CLEAN, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            segments = data if isinstance(data, list) else data.get("segments", data)
            for seg in segments:
                seg["caliber_code"] = caliber_code
                all_segments.append(seg)
        except Exception as e:
            print(f"ERROR loading {fname}: {e}")

    segment_table = pd.DataFrame(all_segments)
    segment_table.columns = [c.lower().strip() for c in segment_table.columns]
    for col in ["duration_sec", "word_count", "speech_rate_wps", "start", "end"]:
        if col in segment_table.columns:
            segment_table[col] = pd.to_numeric(segment_table[col], errors="coerce")
    segment_table["speaker"] = segment_table["speaker"].str.strip().str.lower()
    segment_table["interview_part"] = (
        segment_table["interview_part"].str.strip().str.lower()
    )
    print(f"Total segments : {len(segment_table):,}")
    print(f"Visits loaded  : {segment_table['caliber_code'].nunique()}")

    # Keep only the segments belonging to the analysis visits.
    analysis_codes = set(normalise_code_column(df_audio_clean["caliber_code"]))
    analysis_segments = segment_table[
        segment_table["caliber_code"].isin(analysis_codes)
    ].copy()
    print(f"Segments in analysis visits: {len(analysis_segments):,}")

    # Write one raw JSON per visit and interview part, dropping the columns that
    # were added during loading so the files match their original schema.
    for part, out_dir in PART_OUTPUT_DIRS.items():
        os.makedirs(out_dir, exist_ok=True)
        part_df = analysis_segments[analysis_segments["interview_part"] == part]
        for code, grp in part_df.groupby("caliber_code"):
            segs = grp.sort_values("start").reset_index(drop=True)
            drop_cols = ["caliber_code", "site", "condition", "patient_id"]
            segs = segs.drop(columns=[c for c in drop_cols if c in segs.columns])
            records = segs.to_dict(orient="records")
            out_path = os.path.join(out_dir, f"{code}_{part}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False, indent=2)
        print(f"{part}: {len(os.listdir(out_dir))} JSON files written")

    # Speaker-labelled dialogue text: consecutive same-speaker segments merged.
    dialogue_dirs = {
        "bdae":                TRANSCRIPT_PART_BDAE_DIALOGUE,
        "emotion":             TRANSCRIPT_PART_EMOTION_DIALOGUE,
        "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_DIALOGUE,
    }
    for part, json_dir in PART_OUTPUT_DIRS.items():
        dialogue_dir = dialogue_dirs[part]
        os.makedirs(dialogue_dir, exist_ok=True)
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as fh:
                segs = json.load(fh)
            lines = []
            for speaker, turns in groupby(segs, key=lambda x: x["speaker"]):
                label = "PATIENT" if speaker == "patient" else "INTERVIEWER"
                text = " ".join(
                    str(t["text"]).strip()
                    for t in turns
                    if str(t.get("text", "")).strip()
                )
                if text:
                    lines.append(f"[{label}] {text}")
            out_path = os.path.join(dialogue_dir, fname.replace(".json", ".txt"))
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(lines))
        print(f"{part} dialogue: {len(os.listdir(dialogue_dir))} files written")

    # Patient-only monologue text: patient segments in time order, concatenated.
    monologue_dirs = {
        "bdae":                TRANSCRIPT_PART_BDAE_MONOLOGUE,
        "emotion":             TRANSCRIPT_PART_EMOTION_MONOLOGUE,
        "clinical_evaluation": TRANSCRIPT_PART_CLINICAL_MONOLOGUE,
    }
    for part, json_dir in PART_OUTPUT_DIRS.items():
        monologue_dir = monologue_dirs[part]
        os.makedirs(monologue_dir, exist_ok=True)
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as fh:
                segs = json.load(fh)
            patient_text = " ".join(
                str(s["text"]).strip()
                for s in sorted(segs, key=lambda x: x.get("start", 0))
                if s.get("speaker") == "patient" and str(s.get("text", "")).strip()
            )
            out_path = os.path.join(monologue_dir, fname.replace(".json", ".txt"))
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(patient_text)
        print(f"{part} monologue: {len(os.listdir(monologue_dir))} files written")

    # Produce accent-kept and accent-stripped cleaned monologues, first collapsing
    # extreme ASR loop artifacts.
    _write_cleaned_monologues()


# Thresholds controlling which back-to-back repeated phrases are treated as ASR
# loop artifacts. Genuine emphatic repetition ("no no no") is preserved.
MIN_RUNS_ANY = 6      # any phrase repeated 6+ times in a row → collapse
MIN_LONG_PHRASE = 5   # phrases of 5+ words ...
MIN_RUNS_LONG = 3     # ... collapse at 3+ repeats
MAX_PHRASE_WORDS = 6  # longest phrase length searched for


def collapse_loops(text, log=None, fname=""):
    """Collapse extreme consecutive loops (ASR artifacts) to a single copy.
    Anonymisation tokens are blanked before matching so runs of identical tags do
    not register as loops, but the original tokens are kept."""
    toks = text.split()
    keys = ["" if re.fullmatch(r"<[A-Z_]+>", t.strip()) else t for t in toks]
    n = len(toks)
    out = []
    i = 0
    while i < n:
        matched = False
        for L in range(min(MAX_PHRASE_WORDS, n - i), 0, -1):
            key_phrase = keys[i:i + L]
            if not any(key_phrase):
                continue
            runs = 1
            j = i + L
            while j + L <= n and keys[j:j + L] == key_phrase:
                runs += 1
                j += L
            if runs >= MIN_RUNS_ANY or (L >= MIN_LONG_PHRASE and runs >= MIN_RUNS_LONG):
                out.extend(toks[i:i + L])
                if log is not None:
                    log.append((fname, " ".join(toks[i:i + L]), runs, L))
                i = j
                matched = True
                break
        if not matched:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def clean_for_linguistic(text):
    """Clean a monologue for linguistic analyses (TFMN, co-occurrence, topics,
    emotion): drop the BOM and anonymisation tags but keep accents intact."""
    text = text.lstrip("﻿")
    text = re.sub(r"<[A-Z_]+>", "", text)
    return re.sub(r" +", " ", text).strip()


def clean_for_speechgraph(text):
    """Clean a monologue for the speech-graph software: as clean_for_linguistic,
    but additionally strip accent marks (paradís → paradis) which that tool
    cannot handle."""
    text = text.lstrip("﻿")
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r" +", " ", text).strip()


def _write_cleaned_monologues():
    """Write loop-collapsed monologue variants per part folder: a "_clean" version
    keeping accents and a "_clean_speechgraph" version with accents removed."""
    collapse_log = []
    for folder in [
        TRANSCRIPT_PART_BDAE_MONOLOGUE,
        TRANSCRIPT_PART_EMOTION_MONOLOGUE,
        TRANSCRIPT_PART_CLINICAL_MONOLOGUE,
    ]:
        clean_linguistic = folder + "_clean"
        clean_speechgraph = folder + "_clean_speechgraph"
        os.makedirs(clean_linguistic, exist_ok=True)
        os.makedirs(clean_speechgraph, exist_ok=True)

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as fh:
                text = fh.read()
            text = collapse_loops(text, log=collapse_log, fname=fname)
            with open(os.path.join(clean_linguistic, fname), "w", encoding="utf-8") as fh:
                fh.write(clean_for_linguistic(text))
            with open(os.path.join(clean_speechgraph, fname), "w", encoding="utf-8") as fh:
                fh.write(clean_for_speechgraph(text))

        print(f"{os.path.basename(folder)}:")
        print(f"  → _clean: {len(os.listdir(clean_linguistic))} files (accents kept)")
        print(f"  → _clean_speechgraph: "
              f"{len(os.listdir(clean_speechgraph))} files (accents removed)")

    if collapse_log:
        files_touched = sorted(set(f for f, *_ in collapse_log))
        print(f"Collapsed {len(collapse_log)} loop(s) across "
              f"{len(files_touched)} file(s).")
    else:
        print("No loops met the collapse threshold.")


# ── Section 5: Anonymisation tags ─────────────────────────────────────────────
# Replace anonymisation placeholders (<PERSON>, <PLACE>, ...) in the monologues
# with a plausible common word in the participant's language, giving natural
# input for multilingual embeddings. Language comes from the cleaned metadata;
# unknown codes fall back to Spanish. Output goes to "_embedding_ready" folders.

# Placeholder → replacement word, per language. "mix" defaults to Spanish.
TAG_REPLACEMENTS = {
    "spanish": {
        "PERSON": "persona", "RELATION": "familiar", "PLACE": "lugar",
        "ADDRESS": "dirección", "NUM": "número", "DATE": "fecha",
        "ORG": "organización", "NATIONALITY": "nacionalidad",
    },
    "catalan": {
        "PERSON": "persona", "RELATION": "familiar", "PLACE": "lloc",
        "ADDRESS": "adreça", "NUM": "número", "DATE": "data",
        "ORG": "organització", "NATIONALITY": "nacionalitat",
    },
    "english": {
        "PERSON": "person", "RELATION": "relative", "PLACE": "place",
        "ADDRESS": "address", "NUM": "number", "DATE": "date",
        "ORG": "organisation", "NATIONALITY": "nationality",
    },
    "mix": {
        "PERSON": "persona", "RELATION": "familiar", "PLACE": "lugar",
        "ADDRESS": "dirección", "NUM": "número", "DATE": "fecha",
        "ORG": "organización", "NATIONALITY": "nacionalidad",
    },
}


def clean_for_embeddings(text, lang):
    """Replace anonymisation tags with a language-appropriate word (unknown tags
    become 'algo'), drop the BOM and collapse repeated spaces."""
    replacements = TAG_REPLACEMENTS.get(lang, TAG_REPLACEMENTS[FALLBACK_LANG])

    def replace_tag(match):
        return replacements.get(match.group(1), "algo")

    text = text.lstrip("﻿")
    text = re.sub(r"<([A-Z_]+)>", replace_tag, text)
    return re.sub(r" {2,}", " ", text).strip()


def run_replace_anonymization_tags():
    # Build a caliber_code → language lookup from the cleaned metadata.
    df_meta = load_dataframe(AUDIO_DATA_CLEAN)
    df_meta["caliber_code"] = normalise_code_column(df_meta["caliber_code"])
    lang_lookup = (
        df_meta[["caliber_code", "language"]]
        .dropna(subset=["language"])
        .drop_duplicates(subset="caliber_code")
        .set_index("caliber_code")["language"]
        .to_dict()
    )
    print(f"  Language lookup built for {len(lang_lookup)} participants")

    task_dirs = {
        "bdae":     TRANSCRIPT_PART_BDAE_MONOLOGUE,
        "emotion":  TRANSCRIPT_PART_EMOTION_MONOLOGUE,
        "clinical": TRANSCRIPT_PART_CLINICAL_MONOLOGUE,
    }
    output_dirs = {
        "bdae":     os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "bdae_embedding_ready"),
        "emotion":  os.path.join(TRANSCRIPT_BY_PART_MONOLOGUE, "emotion_embedding_ready"),
        "clinical": os.path.join(
            TRANSCRIPT_BY_PART_MONOLOGUE, "clinical_evaluation_embedding_ready"
        ),
    }

    for task, src_dir in task_dirs.items():
        out_dir = Path(output_dirs[task])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(Path(src_dir).glob("*.txt"))
        lang_counts = Counter()
        changed = 0
        no_lang = []

        for fpath in files:
            code = (
                fpath.stem
                .replace(f"_{task}", "")
                .replace("_bdae", "")
                .replace("_emotion", "")
                .replace("_clinical", "")
                .strip()
                .lower()
            )
            lang = lang_lookup.get(code, FALLBACK_LANG)
            if code not in lang_lookup:
                no_lang.append(fpath.name)

            text = read_text_file(fpath)
            cleaned = clean_for_embeddings(text, lang)
            lang_counts[lang] += 1
            (out_dir / fpath.name).write_text(cleaned, encoding="utf-8")
            if cleaned != text.strip():
                changed += 1

        print(f"\n{task.upper()} → {os.path.basename(output_dirs[task])}/")
        print(f"  Files written  : {len(files)}")
        print(f"  Files modified : {changed}")
        print(f"  Languages used : {dict(lang_counts.most_common())}")
        if no_lang:
            print(f"  No language found (used fallback) for {len(no_lang)} files")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_data_preparation():
    """Run all five preparation stages in order."""
    run_data_cleaning()
    run_metadata_labelling()
    run_unique_patient_selection()
    run_transcript_identification()
    run_replace_anonymization_tags()


if __name__ == "__main__":
    run_data_preparation()
