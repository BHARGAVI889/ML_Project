"""
Blocking / candidate generation — scalable version.

At this dataset's scale (millions of S1 rows, millions of S2/S3 rows),
brute-force pairwise comparison — including TF-IDF cosine similarity via
sklearn's NearestNeighbors, which is exact/exhaustive under the hood — is
NOT viable: it's an O(n * m) matrix computation per country, and a single
country group of a few hundred thousand rows on each side already means
tens of billions of comparisons. That's what was hanging the pipeline.

Instead we use classic record-linkage BLOCKING KEYS: cheap, deterministic
functions of the normalized name that partition records into small
buckets, so we only ever compare records that land in the same bucket.
Bucket construction is a single pass over the data (dict/hashmap), i.e.
O(n), not O(n*m).

Keys used (union of all three, each computed within country):
  1. name_prefix  -> first `prefix_len` chars of the normalized name
                     (catches near-identical names / minor suffix noise)
  2. first_token  -> first whitespace-delimited token of the normalized
                     name (catches reordered/truncated names that still
                     start the same way)
  3. rare_token   -> any token (len >= min_token_len) that is NOT
                     extremely common (document frequency <= max_df on
                     the "other" side) — a safety net for names that
                     share a distinctive word but not a prefix/first token

Any bucket whose "other" side exceeds `max_block_size` is dropped
entirely rather than kept: a bucket that large has essentially zero
discriminative power (e.g. hundreds of businesses sharing a common
prefix), so keeping it only costs time/memory downstream for no recall
benefit. This is what makes the whole thing scale — tune
`max_block_size` down if a country's buckets are still too large (watch
the printed avg/max candidates-per-entity diagnostic).
"""
from collections import defaultdict

from data_utils import normalize_name, token_set


def _prefix_key(name: str, n: int) -> str:
    return name[:n] if name else ""


def _first_token_key(name: str) -> str:
    return name.split()[0] if name else ""


def _build_bucket_index(other_ids, other_keys, max_block_size):
    """other_ids/other_keys: parallel lists. Returns {key: [other_id, ...]},
    dropping any bucket bigger than max_block_size."""
    buckets = defaultdict(list)
    for oid, key in zip(other_ids, other_keys):
        if key:
            buckets[key].append(oid)
    return {k: v for k, v in buckets.items() if len(v) <= max_block_size}


def _apply_bucket_index(s1_ids, s1_keys, bucket_index, candidates):
    for sid, key in zip(s1_ids, s1_keys):
        hits = bucket_index.get(key)
        if hits:
            candidates[sid].update(hits)


def generate_candidates(
    s1_df, other_df, other_source_label,
    prefix_len=4, min_token_len=4, max_block_size=300,
):
    """
    Generate candidate matches from `other_df` (source2 or source3) for every
    row in `s1_df`, grouped by country, using key-based blocking.

    Returns: (dict {source1_entity_id: set(other_entity_id)}, other_source_label)
    """
    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["_norm_name"] = s1_df["business_name"].map(normalize_name)
    other_df["_norm_name"] = other_df["business_name"].map(normalize_name)

    candidates = defaultdict(set)

    for country, s1_group in s1_df.groupby("country"):
        other_group = other_df[other_df["country"] == country]
        if other_group.empty:
            continue

        s1_ids = s1_group["entity_id"].tolist()
        s1_names = s1_group["_norm_name"].tolist()
        other_ids = other_group["entity_id"].tolist()
        other_names = other_group["_norm_name"].tolist()

        # 1) name-prefix blocking
        s1_prefix = [_prefix_key(n, prefix_len) for n in s1_names]
        other_prefix = [_prefix_key(n, prefix_len) for n in other_names]
        idx = _build_bucket_index(other_ids, other_prefix, max_block_size)
        _apply_bucket_index(s1_ids, s1_prefix, idx, candidates)

        # 2) first-token blocking
        s1_first = [_first_token_key(n) for n in s1_names]
        other_first = [_first_token_key(n) for n in other_names]
        idx = _build_bucket_index(other_ids, other_first, max_block_size)
        _apply_bucket_index(s1_ids, s1_first, idx, candidates)

        # 3) rare-token overlap safety net (exploded: one entity can hit
        #    several buckets, one per qualifying token in its name)
        other_tok_buckets = defaultdict(list)
        for oid, name in zip(other_ids, other_names):
            for tok in token_set(name):
                if len(tok) >= min_token_len:
                    other_tok_buckets[tok].append(oid)
        other_tok_buckets = {
            tok: ids for tok, ids in other_tok_buckets.items()
            if len(ids) <= max_block_size
        }
        for sid, name in zip(s1_ids, s1_names):
            for tok in token_set(name):
                if len(tok) >= min_token_len:
                    hits = other_tok_buckets.get(tok)
                    if hits:
                        candidates[sid].update(hits)

    # Ensure every s1 entity has an entry (possibly empty)
    for s1_id in s1_df["entity_id"]:
        candidates.setdefault(s1_id, set())

    sizes = [len(v) for v in candidates.values()]
    total = sum(sizes)
    print(f"[blocking:{other_source_label}] {len(candidates)} S1 entities -> "
          f"{total} candidate pairs (avg {total / max(len(sizes), 1):.1f}/entity, "
          f"max {max(sizes) if sizes else 0}/entity)")

    return candidates, other_source_label
