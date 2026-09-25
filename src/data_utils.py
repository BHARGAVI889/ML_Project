"""
Data loading and normalization utilities for the Business Entity Resolution challenge.
"""
import re
import pandas as pd


LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "ltd", "limited", "llc",
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llp", "plc", "gmbh", "sa", "sas", "pte", "pte ltd",
]

ADDRESS_ABBR = {
    r"\broad\b": "rd", r"\bstreet\b": "st", r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd", r"\blane\b": "ln", r"\bdrive\b": "dr",
    r"\bnear\b": "", r"\bopposite\b": "", r"\bopp\.?\b": "",
}


def load_source(path: str) -> pd.DataFrame:
    """Load a *_sourceN.tsv file. Always use sep='\\t'."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse legal suffixes/abbreviations."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in LEGAL_SUFFIXES:
        s = re.sub(rf"\b{re.escape(suf)}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(addr: str) -> str:
    if not isinstance(addr, str):
        return ""
    s = addr.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    for pattern, repl in ADDRESS_ABBR.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(s: str) -> set:
    return set(s.split()) if s else set()
