"""
Pairwise feature engineering between a Source-1 record and a candidate
Source-2/Source-3 record.
"""
import Levenshtein
from rapidfuzz import fuzz

from data_utils import normalize_name, normalize_address, token_set


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))


def pair_features(s1_row, other_row) -> dict:
    n1 = normalize_name(s1_row["business_name"])
    n2 = normalize_name(other_row["business_name"])
    a1 = normalize_address(s1_row["business_address"])
    a2 = normalize_address(other_row["business_address"])

    n1_tokens, n2_tokens = token_set(n1), token_set(n2)
    a1_tokens, a2_tokens = token_set(a1), token_set(a2)

    feats = {
        "name_lev_ratio": _lev_ratio(n1, n2),
        "name_jaccard": _jaccard(n1_tokens, n2_tokens),
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_len_diff": abs(len(n1) - len(n2)),
        "addr_lev_ratio": _lev_ratio(a1, a2),
        "addr_jaccard": _jaccard(a1_tokens, a2_tokens),
        "addr_token_sort_ratio": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "addr_common_token_count": len(a1_tokens & a2_tokens),
        "country_match": int(
            str(s1_row["country"]).strip().lower() == str(other_row["country"]).strip().lower()
        ),
        # a crude proxy for shared numeric tokens (house/PIN numbers) in the address
        "addr_shared_digits": len(
            {t for t in a1_tokens if t.isdigit()} & {t for t in a2_tokens if t.isdigit()}
        ),
    }
    return feats


FEATURE_COLUMNS = [
    "name_lev_ratio", "name_jaccard", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_diff",
    "addr_lev_ratio", "addr_jaccard", "addr_token_sort_ratio",
    "addr_common_token_count", "country_match", "addr_shared_digits",
]
