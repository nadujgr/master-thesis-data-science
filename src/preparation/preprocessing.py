"""
Transcript preprocessing pipeline for the CALIBER Audio Study. From the raw
Whisper-style JSON transcripts it fixes a few misnamed files, then segments every
transcript by interview part (BDAE, emotion, clinical_evaluation), restricting to
the visits in the cleaned audio metadata. From the segmented data it produces
turn-by-turn dialogue text, patient-only monologue text, and three cleaned
monologue variants: accents preserved for linguistic analysis, accents stripped
for the speech-graph software, and anonymisation tags replaced by generic terms
for embedding models. Inputs are the raw JSON transcripts and the cleaned audio
metadata CSV; outputs are the segmented JSONs and text exports under the
processed-data directories from config.
"""

import json
import os
import re
import shutil
import unicodedata
from itertools import groupby

import pandas as pd

from src.config import (
    AUDIO_DATA_CLEAN,
    TRANSCRIPT_JSON_DIR,
    TRANSCRIPT_JSON_DIR_CLEAN,
    TRANSCRIPT_BY_PART,
    DATA_PROCESSED,
)

# ── Step 1: Fix JSON filenames ────────────────────────────────────────────────

MANUAL_ID_FIXES = {
    "275_mania_bd_acute": "275_me_acute",
    "107_me_resp": "107_me_acute",
    "136_mde_bd_respremi": "136_eu_bd",
    "159_mde_bd_resp": "159_mde_bd_acute",
    "205_mde_bd_respremi": "205_eu_bd",
    "208_eu_bd": "208_mde_bd_acute",
    "264_mde_bd_resp": "264_mde_bd_acute",
    "268_me_bd_acute": "268_me_acute",
    "269_me_bd_acute": "269_me_acute",
    "269_me_bd_resp": "269_me_resp",
}


def fix_json_filenames():
    """Copy JSONs to clean directory with fixed names."""
    print("STEP 1: Fixing JSON filenames")
    print("─" * 70)
    
    os.makedirs(TRANSCRIPT_JSON_DIR_CLEAN, exist_ok=True)
    
    json_files = sorted(f for f in os.listdir(TRANSCRIPT_JSON_DIR) if f.endswith(".json"))
    n_renamed = 0
    
    for fname in json_files:
        orig_name = fname.replace(".json", "")
        fixed_name = MANUAL_ID_FIXES.get(orig_name, orig_name)
        
        src = os.path.join(TRANSCRIPT_JSON_DIR, fname)
        dst = os.path.join(TRANSCRIPT_JSON_DIR_CLEAN, fixed_name + ".json")
        
        shutil.copy2(src, dst)
        if orig_name != fixed_name:
            n_renamed += 1
            print(f"  Renamed: {orig_name} → {fixed_name}")
    
    print(f"✓ {len(json_files)} files processed, {n_renamed} renamed\n")


# ── Step 2: Load and segment by interview part ────────────────────────────────

def load_all_segments():
    """Load all JSON files into flat DataFrame."""
    print("STEP 2: Loading segments")
    print("─" * 70)
    
    all_segments = []
    for fname in sorted(f for f in os.listdir(TRANSCRIPT_JSON_DIR_CLEAN) if f.endswith(".json")):
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
            print(f"  ERROR loading {fname}: {e}")
    
    df = pd.DataFrame(all_segments)
    df.columns = [c.lower().strip() for c in df.columns]

    for col in ["duration_sec", "word_count", "speech_rate_wps", "start", "end"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["speaker"] = df["speaker"].str.strip().str.lower()
    df["interview_part"] = df["interview_part"].str.strip().str.lower()
    
    print(f"✓ {len(df):,} segments from {df['caliber_code'].nunique()} visits")
    print(f"  Parts: {sorted(df['interview_part'].unique())}\n")
    
    return df


def filter_to_analysis_visits(df):
    """Keep only visits in audio_data_clean.csv."""
    df_audio = pd.read_csv(AUDIO_DATA_CLEAN)
    analysis_codes = set(df_audio["caliber_code"].str.strip().str.lower())
    
    df_filtered = df[df["caliber_code"].isin(analysis_codes)].copy()
    print(f"  Filtered to {df_filtered['caliber_code'].nunique()} analysis visits")
    print(f"  ({len(df_filtered):,} segments)\n")
    
    return df_filtered


# ── Step 3: Export by interview part (JSON) ───────────────────────────────────

def export_by_part_json(df):
    """Export segmented JSONs by interview part."""
    print("STEP 3: Exporting by interview part (JSON)")
    print("─" * 70)
    
    parts = {
        "bdae": os.path.join(TRANSCRIPT_BY_PART, "bdae"),
        "emotion": os.path.join(TRANSCRIPT_BY_PART, "emotion"),
        "clinical_evaluation": os.path.join(TRANSCRIPT_BY_PART, "clinical_evaluation"),
    }
    
    for part, out_dir in parts.items():
        os.makedirs(out_dir, exist_ok=True)
        part_df = df[df["interview_part"] == part]
        
        for code, grp in part_df.groupby("caliber_code"):
            segs = grp.sort_values("start").reset_index(drop=True)
            drop_cols = ["caliber_code", "site", "condition", "patient_id"]
            segs = segs.drop(columns=[c for c in drop_cols if c in segs.columns])
            
            records = segs.to_dict(orient="records")
            out_path = os.path.join(out_dir, f"{code}_{part}.json")
            
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False, indent=2)
        
        print(f"  {part}: {len(os.listdir(out_dir))} files")
    print()


# ── Step 4: Export dialogue format (turn-by-turn) ─────────────────────────────

def export_dialogue_format(df):
    """Export turn-by-turn dialogue with speaker labels."""
    print("STEP 4: Exporting dialogue format")
    print("─" * 70)
    
    base_dir = os.path.join(DATA_PROCESSED, "transcript_by_part_dialogue")
    parts = {
        "bdae": os.path.join(base_dir, "bdae"),
        "emotion": os.path.join(base_dir, "emotion"),
        "clinical_evaluation": os.path.join(base_dir, "clinical_evaluation"),
    }
    
    for part, out_dir in parts.items():
        os.makedirs(out_dir, exist_ok=True)
        json_dir = os.path.join(TRANSCRIPT_BY_PART, part)
        
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            
            with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as fh:
                segs = json.load(fh)
            
            lines = []
            for speaker, turns in groupby(segs, key=lambda x: x["speaker"]):
                label = "PATIENT" if speaker == "patient" else "INTERVIEWER"
                text = " ".join(
                    str(t["text"]).strip() for t in turns
                    if str(t.get("text", "")).strip()
                )
                if text:
                    lines.append(f"[{label}] {text}")
            
            content = "\n\n".join(lines)
            out_path = os.path.join(out_dir, fname.replace(".json", ".txt"))
            
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(content)
        
        print(f"  {part}: {len(os.listdir(out_dir))} files")
    print()


# ── Step 5: Export monologue format (patient-only) ────────────────────────────

def check_text_quality(text, filename):
    """Check concatenated patient speech for common errors; return list of
    warnings."""
    warnings = []

    # Missing sentence boundaries: lowercase followed by uppercase.
    missing_punct = re.findall(r'[a-z]\s+[A-Z]', text)
    if missing_punct and len(missing_punct) > 3:
        warnings.append(f"Missing punctuation ({len(missing_punct)} cases)")

    # Word repeated 3+ times.
    repetitions = re.findall(r'\b(\w+)(\s+\1){2,}\b', text)
    if repetitions:
        warnings.append(f"Repetitions: {len(repetitions)} instances")

    # Encoding errors (replacement character).
    if '�' in text:
        count = text.count('�')
        warnings.append(f"Encoding errors: {count} × �")

    # Anonymization tokens left.
    anon_tokens = re.findall(r'<[A-Z_]+>', text)
    if anon_tokens:
        warnings.append(f"Anonymization tokens: {set(anon_tokens)}")

    word_count = len(text.split())
    if word_count < 20:
        warnings.append(f"Short output: {word_count} words")
    
    if warnings:
        print(f"    ⚠ {filename}: {'; '.join(warnings)}")
    
    return warnings


def export_monologue_format(df):
    """Export patient-only monologue (original, uncleaned)."""
    print("STEP 5: Exporting monologue format (original)")
    print("─" * 70)
    
    base_dir = os.path.join(DATA_PROCESSED, "transcript_by_part_monologue")
    parts = {
        "bdae": os.path.join(base_dir, "bdae"),
        "emotion": os.path.join(base_dir, "emotion"),
        "clinical_evaluation": os.path.join(base_dir, "clinical_evaluation"),
    }
    
    all_warnings = []
    
    for part, out_dir in parts.items():
        os.makedirs(out_dir, exist_ok=True)
        json_dir = os.path.join(TRANSCRIPT_BY_PART, part)
        
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            
            with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as fh:
                segs = json.load(fh)
            
            # Patient segments only, sorted by start time.
            patient_text = " ".join(
                str(s["text"]).strip()
                for s in sorted(segs, key=lambda x: x.get("start", 0))
                if s.get("speaker") == "patient"
                and str(s.get("text", "")).strip()
            )
            
            warnings = check_text_quality(patient_text, fname)
            if warnings:
                all_warnings.append((fname, warnings))
            
            out_path = os.path.join(out_dir, fname.replace(".json", ".txt"))
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(patient_text)
        
        print(f"  {part}: {len(os.listdir(out_dir))} files")
    
    if all_warnings:
        print(f"\n  Total files with warnings: {len(all_warnings)}")
    print()


# ── Step 6: Cleaning (linguistic / speech graphs / embeddings) ────────────────

# Tag replacements for the embedding-ready version, per language.
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
}


def clean_for_linguistic_analysis(text):
    """Minimal cleaning for TFMN, co-occurrence, topics, emotion; preserves
    accents. Removes BOM and anonymisation tags, adds sentence-boundary
    punctuation, and drops adjacent word repetitions."""
    text = text.lstrip('\ufeff')
    text = re.sub(r'<[A-Z_]+>', '', text)

    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    text = re.sub(r'([a-z])\s+(¿[A-Z])', r'\1. \2', text)

    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    text = re.sub(r' +', ' ', text).strip()

    return text


def clean_for_speechgraph(text):
    """Cleaning for the speech-graph software: as clean_for_linguistic_analysis,
    but additionally strips accents (required by the software)."""
    text = text.lstrip('\ufeff')
    text = re.sub(r'<[A-Z_]+>', '', text)

    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    text = re.sub(r'([a-z])\s+(¿[A-Z])', r'\1. \2', text)

    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    # Strip accents: paradís → paradis, año → ano.
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    text = re.sub(r' +', ' ', text).strip()

    return text


def clean_for_embeddings(text, lang):
    """Replace anonymisation tags with language-appropriate generic terms for
    embedding models, then normalise punctuation and spacing."""
    replacements = TAG_REPLACEMENTS.get(lang.lower(), TAG_REPLACEMENTS.get("spanish"))

    def replace_tag(match):
        return replacements.get(match.group(1), "algo")

    text = text.lstrip('\ufeff')
    text = re.sub(r'<([A-Z_]+)>', replace_tag, text)

    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    text = re.sub(r'([a-z])\s+(¿[A-Z])', r'\1. \2', text)

    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    text = re.sub(r' +', ' ', text).strip()

    return text


def dual_clean_monologues():
    """Create three cleaned versions: linguistic, speechgraph, and embedding-ready."""
    print("STEP 6: Triple cleaning (linguistic / speech graphs / embeddings)")
    print("─" * 70)

    df_meta = pd.read_csv(AUDIO_DATA_CLEAN)
    df_meta["caliber_code"] = df_meta["caliber_code"].astype(str).str.strip().str.lower()
    lang_lookup = (
        df_meta[["caliber_code", "language"]]
        .dropna(subset=["language"])
        .drop_duplicates(subset="caliber_code")
        .set_index("caliber_code")["language"]
        .str.lower()
        .to_dict()
    )
    
    base_dir = os.path.join(DATA_PROCESSED, "transcript_by_part_monologue")
    
    parts = ["bdae", "emotion", "clinical_evaluation"]
    
    for part in parts:
        orig_dir = os.path.join(base_dir, part)
        clean_ling = os.path.join(base_dir, f"{part}_clean")
        clean_sg = os.path.join(base_dir, f"{part}_clean_speechgraph")
        clean_emb = os.path.join(base_dir, f"{part}_embedding_ready")
        
        os.makedirs(clean_ling, exist_ok=True)
        os.makedirs(clean_sg, exist_ok=True)
        os.makedirs(clean_emb, exist_ok=True)
        
        for fname in sorted(os.listdir(orig_dir)):
            if not fname.endswith(".txt"):
                continue

            code = (fname.replace(".txt", "")
                    .replace(f"_{part}", "")
                    .replace("_bdae", "")
                    .replace("_emotion", "")
                    .replace("_clinical_evaluation", "")
                    .strip().lower())
            
            lang = lang_lookup.get(code, "spanish")
            
            with open(os.path.join(orig_dir, fname), "r", encoding="utf-8") as fh:
                text = fh.read()
            
            # Linguistic: accents preserved, tags removed.
            text_ling = clean_for_linguistic_analysis(text)
            with open(os.path.join(clean_ling, fname), "w", encoding="utf-8") as fh:
                fh.write(text_ling)

            # Speech graphs: accents removed, tags removed.
            text_sg = clean_for_speechgraph(text)
            with open(os.path.join(clean_sg, fname), "w", encoding="utf-8") as fh:
                fh.write(text_sg)

            # Embeddings: accents preserved, tags replaced.
            text_emb = clean_for_embeddings(text, lang)
            with open(os.path.join(clean_emb, fname), "w", encoding="utf-8") as fh:
                fh.write(text_emb)
        
        print(f"  {part}:")
        print(f"    → {os.path.basename(clean_ling)}: {len(os.listdir(clean_ling))} files")
        print(f"    → {os.path.basename(clean_sg)}: {len(os.listdir(clean_sg))} files")
        print(f"    → {os.path.basename(clean_emb)}: {len(os.listdir(clean_emb))} files")
    
    print("\n  USAGE:")
    print("    Speech graphs:  use *_clean_speechgraph/")
    print("    TFMN/topics:    use *_clean/")
    print("    Embeddings:     use *_embedding_ready/")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocessing_transcripts():
    """Run the full preprocessing pipeline end to end: fix filenames, load and
    filter segments, then export the JSON, dialogue, monologue, and cleaned
    monologue variants in sequence."""
    print("═" * 70)
    print("TRANSCRIPT PREPROCESSING PIPELINE")
    print("═" * 70)
    print()

    fix_json_filenames()

    df = load_all_segments()
    df = filter_to_analysis_visits(df)

    export_by_part_json(df)
    export_dialogue_format(df)
    export_monologue_format(df)
    dual_clean_monologues()

    print("═" * 70)
    print("✓ PREPROCESSING COMPLETE")
    print("═" * 70)
    print("\nOutput structure:")
    print("  transcript_by_part/              → Segmented JSONs")
    print("  transcript_by_part_dialogue/     → Turn-by-turn TXT")
    print("  transcript_by_part_monologue/")
    print("    bdae/                          → Original patient speech")
    print("    bdae_clean/                    → TFMN/topics (accents, tags removed)")
    print("    bdae_clean_speechgraph/        → Speech graphs (no accents, tags removed)")
    print("    bdae_embedding_ready/          → Embeddings (accents, tags replaced)")
    print("    [same for emotion, clinical_evaluation]")


if __name__ == "__main__":
    preprocessing_transcripts()