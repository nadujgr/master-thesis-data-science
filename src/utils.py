"""
Shared I/O and filename-normalisation helpers for the analysis pipeline.

This module collects small utility functions that were previously duplicated,
byte-for-byte, across several of the analysis scripts (s20, s23, s25, s27, …).
It holds a fallback-encoding text reader, a numpy/pandas -> JSON-safe converter,
a filename -> caliber_code parser, and two caliber_code normalisers (one for a
pandas Series, one for a single string). Behaviour is identical to the original
per-script copies; the functions were merely lifted here so the scripts can
import one shared implementation instead of each carrying its own.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import re
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Local ────────────────────────────────────────────────────────────────────
# (no local imports)


def read_text_with_fallback(
    path,
    encodings=("utf-8", "windows-1252", "latin-1"),
) -> str:
    """Read a text file, trying each encoding in turn until one succeeds.

    Each encoding in ``encodings`` is attempted in order; the first that decodes
    cleanly returns the file's stripped text. Encodings that raise
    ``UnicodeDecodeError`` or ``LookupError`` are skipped. If every candidate
    fails, the file is read one last time as UTF-8 with ``errors="replace"`` so
    the call never raises on undecodable bytes. The default encoding chain is
    the one shared by most call sites; sites needing a different chain pass an
    explicit ``encodings`` argument to preserve their original behaviour.
    """
    for enc in encodings:
        try:
            return Path(path).read_text(encoding=enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()


def json_safe(obj):
    """Recursively convert numpy/pandas objects into JSON-serialisable types.

    Dicts and lists are walked element-by-element; numpy scalar types
    (``np.integer``, ``np.floating``, ``np.bool_``) are cast to their Python
    equivalents; ``np.ndarray`` and ``pd.Series`` become lists and
    ``pd.DataFrame`` becomes a list of record dicts. Anything else is returned
    unchanged, making the result safe to pass to ``json.dump``.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def parse_code(filename: str) -> str:
    """Extract a caliber_code from a filename by stripping known suffixes.

    Takes the file stem and removes a trailing ``_emotion``,
    ``_clinical_evaluation`` or ``_bdae`` marker if present, then lower-cases
    and strips surrounding whitespace. Filenames without one of those suffixes
    yield the bare stem, lower-cased and stripped.
    """
    stem = Path(filename).stem
    for suffix in ("_emotion", "_clinical_evaluation", "_bdae"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)].strip().lower()
    return stem.strip().lower()


def clean_caliber_code_str(raw: str, task: str) -> str:
    """Normalise a single filename string into a caliber_code.

    Lower-cases and strips the input, drops everything from a ``_clean`` marker
    onward, and removes a trailing ``_emotion``/``_bdae``/``_clinical_evaluation``
    suffix. The ``task`` argument is accepted for call-site compatibility but is
    not used, matching the original per-script implementation.
    """
    cleaned = raw.lower().strip()
    cleaned = re.sub(r"_clean.*", "", cleaned)
    cleaned = re.sub(r"_(emotion|bdae|clinical_evaluation)$", "", cleaned)
    return cleaned


def clean_caliber_code_series(s: pd.Series) -> pd.Series:
    """Normalise a pandas Series of filenames into caliber_codes.

    Vectorised counterpart to the string variant: drops everything from a
    ``_random`` marker onward, removes a trailing
    ``_emotion``/``_bdae``/``_clinical_evaluation`` suffix, then strips and
    lower-cases each entry. Note this uses the ``_random`` prefix rule (not
    ``_clean``), matching the original speech-graphs implementation.
    """
    return (s.str.replace(r"_random.*", "", regex=True)
             .str.replace(r"_(emotion|bdae|clinical_evaluation)$", "", regex=True)
             .str.strip().str.lower())
