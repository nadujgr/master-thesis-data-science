#!/usr/bin/env python3
"""
Snippet-based language detection for the uncleaned monologue transcripts. Each
transcript is split into overlapping 25-word snippets, langdetect labels each
snippet, and the majority vote gives the dominant language. Prints diagnostics
(agreement with clinical record labels, cross-task consistency, within-transcript
and patient-level mixing, per-task distributions) and then writes the best
per-code language label back into the clinical CSVs.

Run from the project root; requires `pip install langdetect`. Accepts optional
--mono-root / --clinical-csv / --longitudinal-csv path overrides.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 42
except ImportError:
    print("ERROR: langdetect not installed.  Run:  pip install langdetect")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mono-root",        default=None)
parser.add_argument("--clinical-csv",     default=None)
parser.add_argument("--longitudinal-csv", default=None)
args, _ = parser.parse_known_args()

BASE      = Path(__file__).resolve().parent.parent
MONO_ROOT = Path(args.mono_root) if args.mono_root else BASE / "data" / "processed" / "transcript_by_part_monologue"
CLIN_CSV  = Path(args.clinical_csv)      if args.clinical_csv      else BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_unique_patients.csv"
LONG_CSV  = Path(args.longitudinal_csv)  if args.longitudinal_csv  else BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_longitudinal.csv"

TASKS    = ["bdae", "emotion", "clinical_evaluation"]
LANG_MAP = {"ca": "catalan", "es": "spanish", "en": "english"}
FALLBACK = "spanish"

SNIPPET_WORDS     = 25
SNIPPET_STEP      = 15
MIN_SNIPPET_CHARS = 20
MIN_WORDS_TOTAL   = 30
MIXING_THRESHOLD  = 0.85   # dominant-language share below which a transcript is "mixed"
MIXING_THRESHOLD_STRICT = 0.80   # stricter secondary threshold for patient-level summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text):
    """Strip transcription markup (e.g. <SILENCE> tags and short bracketed
    annotations) and collapse runs of whitespace, returning plain text suitable
    for language detection."""
    text = re.sub(r"<[A-Z_]+>", " ", text)
    text = re.sub(r"\[[^\]]{1,30}\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_snippets(text):
    """Split the cleaned transcript into overlapping word windows and run
    langdetect on each, returning the list of detected language labels (mapped
    to canonical names). Transcripts shorter than MIN_WORDS_TOTAL return an empty
    list; snippets shorter than MIN_SNIPPET_CHARS or that fail detection are
    skipped."""
    cleaned = clean_text(text)
    words   = cleaned.split()
    if len(words) < MIN_WORDS_TOTAL:
        return []
    labels = []
    for i in range(0, len(words), SNIPPET_STEP):
        snippet = " ".join(words[i : i + SNIPPET_WORDS])
        if len(snippet) < MIN_SNIPPET_CHARS:
            continue
        try:
            lang = detect(snippet)
            labels.append(LANG_MAP.get(lang, FALLBACK))
        except LangDetectException:
            pass
    return labels


def dominant_language(labels):
    """Return the majority-vote language across a list of snippet labels together
    with its proportion (rounded to three decimals). Falls back to FALLBACK with
    proportion 0.0 when the list is empty."""
    if not labels:
        return FALLBACK, 0.0
    counts = Counter(labels)
    dom, n = counts.most_common(1)[0]
    return dom, round(n / len(labels), 3)


def code_from_stem(stem, task):
    """Recover the bare caliber_code from a transcript filename stem by removing
    the task suffix and any trailing '_clean...' marker."""
    code = stem.lower().strip()
    for suffix in [f"_{task}", "_bdae", "_emotion", "_clinical_evaluation"]:
        code = code.replace(suffix, "")
    return re.sub(r"_clean.*", "", code)


# ── Load clinical record labels ───────────────────────────────────────────────
clinical_labels = {}
if HAS_PANDAS:
    dfs = []
    for csv_path, label in [(CLIN_CSV, "unique"), (LONG_CSV, "longitudinal")]:
        if csv_path.exists():
            df_tmp = pd.read_csv(csv_path)
            df_tmp["caliber_code"] = df_tmp["caliber_code"].astype(str).str.lower().str.strip()
            if "language" in df_tmp.columns:
                dfs.append(df_tmp[["caliber_code", "language"]].dropna(subset=["language"]))
                print(f"  Loaded {len(df_tmp)} records from {csv_path.name} ({label})")
        else:
            print(f"  ⚠ Not found: {csv_path.name} ({label})")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="caliber_code")
        clinical_labels = (
            combined.set_index("caliber_code")["language"]
            .str.lower().str.strip()
            .to_dict()
        )
        print(f"  Combined: {len(clinical_labels)} unique caliber_codes with language labels")
else:
    print("  No pandas — skipping record-label comparison.")

# ── Step 1: Snippet-based detection (text discarded after each file) ──────────

print("\n" + "=" * 70)
print("  SNIPPET-BASED LANGUAGE DETECTION  (25-word windows, step 15)")
print("=" * 70)

# {caliber_code: {task: {dominant, pct, n_snippets, counts, short, n_words}}}
snippet_results = defaultdict(dict)

for task in TASKS:
    task_dir = MONO_ROOT / task
    if not task_dir.exists():
        print(f"  ⚠  {task_dir} not found — skipping '{task}'")
        continue

    files = sorted(task_dir.glob("*.txt"))
    print(f"\n  {task}: {len(files)} files", flush=True)

    for fpath in files:
        code = code_from_stem(fpath.stem, task)
        try:
            text = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fpath.read_text(encoding="windows-1252", errors="replace")

        n_words = len(text.split())
        labels  = detect_snippets(text)
        del text

        if not labels:
            snippet_results[code][task] = {
                "dominant":   FALLBACK,
                "pct":        1.0,
                "n_snippets": 0,
                "counts":     Counter({FALLBACK: 1}),
                "short":      True,
                "n_words":    n_words,
            }
        else:
            dom, pct = dominant_language(labels)
            snippet_results[code][task] = {
                "dominant":   dom,
                "pct":        pct,
                "n_snippets": len(labels),
                "counts":     Counter(labels),
                "short":      False,
                "n_words":    n_words,
            }

print(f"\n  Total unique caliber_codes: {len(snippet_results)}", flush=True)


# ── Step 2: Snippet label vs clinical record label ────────────────────────────

print("\n" + "=" * 70)
print("  SNIPPET LABEL vs CLINICAL RECORD LABEL")
print("=" * 70)

if clinical_labels:
    agree = disagree = no_record = 0
    disagree_detail = []

    for code, tm in sorted(snippet_results.items()):
        all_doms = [v["dominant"] for v in tm.values()]
        pat_dom  = Counter(all_doms).most_common(1)[0][0]
        rec_lang = clinical_labels.get(code, "")
        if not rec_lang:
            no_record += 1
        elif rec_lang == pat_dom:
            agree += 1
        else:
            disagree += 1
            per_task = "  ".join(
                f"{t}={tm[t]['dominant']}({tm[t]['pct']:.0%})"
                for t in TASKS if t in tm
            )
            disagree_detail.append((code, rec_lang, pat_dom, per_task))

    print(f"\n  Agreement:    {agree}")
    print(f"  Disagreement: {disagree}")
    print(f"  No record:    {no_record}")

    if disagree_detail:
        print(f"\n  Disagreements (record → snippet):")
        print(f"  {'Code':<25}  {'Record':<10}  {'Snippet':<10}  Per-task")
        print("  " + "-" * 80)
        for code, rec, snip, per_task in disagree_detail:
            print(f"  {code:<25}  {rec:<10}  {snip:<10}  {per_task}")
else:
    print("\n  (Skipped — no clinical record labels loaded.)")


# ── Step 3: Cross-task language consistency ───────────────────────────────────

print("\n" + "=" * 70)
print("  CROSS-TASK LANGUAGE CONSISTENCY  (snippet-dominant per task)")
print("=" * 70)

consistent   = []
inconsistent = []

for code, tm in sorted(snippet_results.items()):
    if len(tm) < 2:
        continue
    doms = {t: v["dominant"] for t, v in tm.items()}
    if len(set(doms.values())) == 1:
        consistent.append(code)
    else:
        inconsistent.append((code, tm))

print(f"\n  Consistent across tasks:   {len(consistent)}")
print(f"  Inconsistent across tasks: {len(inconsistent)}")

if inconsistent:
    print(f"\n  {'Code':<25}  {'bdae':<22}  {'emotion':<22}  {'clinical_evaluation'}")
    print("  " + "-" * 92)
    for code, tm in inconsistent:
        def fmt(task):
            if task not in tm:
                return "—"
            r = tm[task]
            return f"{r['dominant']} ({r['pct']:.0%}, n={r['n_snippets']})"
        print(f"  {code:<25}  {fmt('bdae'):<22}  {fmt('emotion'):<22}  {fmt('clinical_evaluation')}")

    print(f"\n  Full snippet breakdown for inconsistent patients:")
    for code, tm in inconsistent:
        print(f"\n  {code}")
        for task in TASKS:
            if task not in tm:
                continue
            r = tm[task]
            counts_str = "  ".join(f"{l}:{n}" for l, n in r["counts"].most_common())
            print(f"    {task:<25}  dominant={r['dominant']}  pct={r['pct']:.0%}  "
                  f"n_snippets={r['n_snippets']}  [{counts_str}]")
else:
    print("\n  ✓ All patients consistent across tasks.")


# ── Step 4: Within-transcript mixing summary ──────────────────────────────────

print("\n" + "=" * 70)
print(f"  WITHIN-TRANSCRIPT MIXING  (dominant snippet < {MIXING_THRESHOLD:.0%})")
print("=" * 70)

mixed = [
    (code, task, v)
    for code, tm in sorted(snippet_results.items())
    for task, v in tm.items()
    if not v["short"] and v["pct"] < MIXING_THRESHOLD
]

n_analysed = sum(
    1 for tm in snippet_results.values()
    for v in tm.values()
    if not v["short"]
)

print(f"\n  Transcripts snippet-analysed (≥{MIN_WORDS_TOTAL} words): {n_analysed}")
print(f"  Transcripts with dominant < {MIXING_THRESHOLD:.0%}:        {len(mixed)}")

if mixed:
    print(f"\n  {'Code':<25}  {'Task':<22}  {'Dominant':<10}  "
          f"{'Pct dom':>8}  {'N snips':>8}  Other")
    print("  " + "-" * 95)
    for code, task, v in sorted(mixed, key=lambda x: x[2]["pct"]):
        other = {k: n for k, n in v["counts"].items() if k != v["dominant"]}
        other_str = "  ".join(f"{k}:{n}" for k, n in sorted(other.items(), key=lambda x: -x[1]))
        print(f"  {code:<25}  {task:<22}  {v['dominant']:<10}  "
              f"{v['pct']:>8.0%}  {v['n_snippets']:>8}  {other_str}")

    print("\n  By task:")
    for task, n in sorted(Counter(task for _, task, _ in mixed).items()):
        print(f"    {task:<25}  {n}")

    print("\n  Mixing patterns (dominant language):")
    for lang, n in Counter(v["dominant"] for _, _, v in mixed).most_common():
        print(f"    dominant={lang}: {n}")
else:
    print(f"\n  ✓ No transcripts below {MIXING_THRESHOLD:.0%} dominance threshold.")

# ── Step 4b: Patient-level mixing summary (across all tasks combined) ─────────

print("\n" + "=" * 70)
print("  PATIENT-LEVEL MIXING SUMMARY  (across all tasks combined)")
print("=" * 70)

# Pool each patient's snippets across all tasks into one distribution
patient_mixing = []
for code, tm in sorted(snippet_results.items()):
    pooled = Counter()
    total_snippets = 0
    for v in tm.values():
        if not v["short"]:
            pooled.update(v["counts"])
            total_snippets += v["n_snippets"]
    
    if total_snippets == 0:
        continue
    
    dom, dom_n = pooled.most_common(1)[0]
    pct_dom = dom_n / total_snippets
    
    patient_mixing.append({
        "code":           code,
        "dominant":       dom,
        "pct_dominant":   round(pct_dom, 3),
        "total_snippets": total_snippets,
        "counts":         dict(pooled),
    })

n_patients_analysed = len(patient_mixing)
n_below_85 = sum(1 for p in patient_mixing if p["pct_dominant"] < MIXING_THRESHOLD)
n_below_80 = sum(1 for p in patient_mixing if p["pct_dominant"] < MIXING_THRESHOLD_STRICT)

print(f"\n  Patients with sufficient transcript length: {n_patients_analysed}")
print(f"  Patients with pooled dominant < 85%:        {n_below_85} "
      f"({100*n_below_85/n_patients_analysed:.1f}%)")
print(f"  Patients with pooled dominant < 80%:        {n_below_80} "
      f"({100*n_below_80/n_patients_analysed:.1f}%)")

if n_below_85 > 0:
    print(f"\n  Patients below 85% dominance threshold:")
    print(f"  {'Code':<25}  {'Dominant':<10}  {'Pct dom':>8}  "
          f"{'N snips':>8}  Other")
    print("  " + "-" * 75)
    for p in sorted(patient_mixing, key=lambda x: x["pct_dominant"]):
        if p["pct_dominant"] < MIXING_THRESHOLD:
            other = {k: n for k, n in p["counts"].items() if k != p["dominant"]}
            other_str = "  ".join(
                f"{k}:{n}" for k, n in sorted(other.items(), key=lambda x: -x[1])
            )
            print(f"  {p['code']:<25}  {p['dominant']:<10}  "
                  f"{p['pct_dominant']:>8.0%}  {p['total_snippets']:>8}  {other_str}")

print(f"\n  Mixing patterns (dominant language, patients below 85%):")
for lang, n in Counter(
    p["dominant"] for p in patient_mixing if p["pct_dominant"] < MIXING_THRESHOLD
).most_common():
    print(f"    dominant={lang}: {n}")

# ── Step 5: Language distribution per task ────────────────────────────────────

print("\n" + "=" * 70)
print("  LANGUAGE DISTRIBUTION PER TASK  (snippet-dominant labels)")
print("=" * 70)

for task in TASKS:
    doms = [tm[task]["dominant"] for tm in snippet_results.values() if task in tm]
    if not doms:
        continue
    counts = Counter(doms)
    total  = len(doms)
    print(f"\n  {task}  (N={total})")
    for lang, n in sorted(counts.items()):
        print(f"    {lang:<10}  {n:>3}  ({100*n/total:.1f}%)")


# ── Step 6: Final label summary ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL LANGUAGE LABEL SUMMARY  (snippet-dominant, majority across tasks)")
print("=" * 70)

all_finals = [
    Counter([v["dominant"] for v in tm.values()]).most_common(1)[0][0]
    for tm in snippet_results.values()
]
print(f"\n  Overall distribution:")
for lang, n in Counter(all_finals).most_common():
    print(f"    {lang:<10}  {n}")


print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)

# ── Step 7: Update language labels in clinical CSVs ───────────────────────────

print("\n" + "=" * 70)
print("  UPDATING LANGUAGE LABELS IN CLINICAL CSVs")
print("=" * 70)

if not HAS_PANDAS:
    print("  ⚠ pandas not available — skipping CSV updates.")
else:
    # Best language label per caliber_code, in priority order:
    #   1. clinical_evaluation dominant (longest, most reliable transcript)
    #   2. majority vote across tasks with real snippets
    #   3. majority vote across all tasks, including short ones
    best_labels: dict[str, str] = {}

    for code, tm in snippet_results.items():
        ce = tm.get("clinical_evaluation")
        if ce and not ce["short"] and ce["n_snippets"] > 0:
            best_labels[code] = ce["dominant"]
            continue

        real_doms = [v["dominant"] for v in tm.values() if not v["short"] and v["n_snippets"] > 0]
        if real_doms:
            best_labels[code] = Counter(real_doms).most_common(1)[0][0]
            continue

        all_doms = [v["dominant"] for v in tm.values()]
        if all_doms:
            best_labels[code] = Counter(all_doms).most_common(1)[0][0]

    print(f"\n  Best labels computed for {len(best_labels)} caliber_codes")
    print(f"  Distribution: {dict(Counter(best_labels.values()).most_common())}")

    # Update each CSV
    CSV_PATHS = [
        BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_unique_patients.csv",
        BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_longitudinal.csv",
        BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_labelled.csv",
        BASE / "data" / "processed" / "caliber_audio" / "caliber_audio_clean.csv",
    ]

    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"\n  ⚠ Not found: {csv_path.name} — skipping")
            continue

        df = pd.read_csv(csv_path)
        df["caliber_code"] = df["caliber_code"].astype(str).str.lower().str.strip()

        if "language" not in df.columns:
            print(f"\n  ⚠ No language column in {csv_path.name} — skipping")
            continue

        original = df["language"].copy()
        df["language"] = df["caliber_code"].map(best_labels).fillna(df["language"])

        n_changed = (df["language"] != original).sum()
        n_filled  = original.isna().sum() - df["language"].isna().sum()

        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ {csv_path.name}")
        print(f"    {n_changed} labels updated, {n_filled} missing values filled")
        print(f"    New distribution: {dict(df['language'].value_counts())}")