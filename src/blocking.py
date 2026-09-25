"""
Blocking / candidate generation.

Strategy (all within-country, since cross-country matches are essentially
impossible for real businesses and this collapses the search space a lot):
  1. TF-IDF character n-gram cosine similarity on normalized business_name
     -> top-K nearest neighbors per Source-1 entity, per candidate source.
     Computed as a sparse matrix product with a per-row top-k extraction,
     so we never materialize a dense N x M similarity matrix in memory —
     that dense-matrix blowup is what causes multi-GB memory use on large
     datasets.
  2. A cheap exact/near-exact token-overlap block as a recall safety net,
     so we don't miss matches TF-IDF ranks low but that share rare tokens
     (e.g. a distinctive brand word). Very common tokens are skipped so a
     single frequent word (e.g. "international") can't blow up every
     candidate set at once.
Both candidate sets are unioned. This union IS what should end up in
candidate_pairs.tsv, since it's the exact set fed to the matching model.
"""
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import normalize_name, token_set

# Cap on TF-IDF vocabulary size per country group. Without this, a large
# corpus of char n-grams (2-4 chars) can create a huge sparse matrix and
# blow up memory during the similarity computation.
MAX_TFIDF_FEATURES = 20_000
# A token that appears in more than this fraction of records is too common
# to be a useful blocking key (and would create huge candidate sets).
MAX_TOKEN_DOC_FREQ_RATIO = 0.02


def _tfidf_topk(s1_names, other_names, top_k=15, row_batch_size=2000):
    """
    Return, for each s1 index, a list of (other_idx, cosine_sim), computed
    via sparse matrix multiplication in row batches so memory stays bounded
    regardless of corpus size.
    """
    if not other_names or not s1_names:
        return [[] for _ in s1_names]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2,
        max_features=MAX_TFIDF_FEATURES, dtype=np.float32,
    )
    all_text = s1_names + other_names
    tfidf = vectorizer.fit_transform(all_text)
    s1_vec = tfidf[: len(s1_names)]
    other_vec = tfidf[len(s1_names):]
    other_vec_t = other_vec.T.tocsr()  # for fast sparse matmul below

    k = min(top_k, other_vec.shape[0])
    results = [[] for _ in range(len(s1_names))]

    for start in range(0, s1_vec.shape[0], row_batch_size):
        end = min(start + row_batch_size, s1_vec.shape[0])
        # sparse (batch x vocab) @ (vocab x other) -> sparse (batch x other)
        sims_batch = (s1_vec[start:end] @ other_vec_t).tocsr()
        for local_i in range(sims_batch.shape[0]):
            row = sims_batch.getrow(local_i)
            if row.nnz == 0:
                continue
            # top-k within this row's nonzero entries only
            if row.nnz > k:
                top_local = np.argpartition(row.data, -k)[-k:]
            else:
                top_local = np.arange(row.nnz)
            cols = row.indices[top_local]
            vals = row.data[top_local]
            results[start + local_i] = list(zip(cols.tolist(), vals.tolist()))

    return results


def _token_block(s1_norm_names, other_norm_names, min_token_len=4):
    """Inverted index on tokens length >= min_token_len -> set of candidate indices sharing a rare-but-not-too-rare token."""
    inv_index = defaultdict(list)
    for j, name in enumerate(other_norm_names):
        for tok in token_set(name):
            if len(tok) >= min_token_len:
                inv_index[tok].append(j)

    # Drop tokens that are too common to be a useful blocking key —
    # these are what cause runaway candidate-set sizes / memory use.
    max_doc_freq = max(5, int(MAX_TOKEN_DOC_FREQ_RATIO * max(len(other_norm_names), 1)))
    inv_index = {tok: ids for tok, ids in inv_index.items() if len(ids) <= max_doc_freq}

    result = []
    for name in s1_norm_names:
        cand = set()
        for tok in token_set(name):
            if len(tok) >= min_token_len:
                cand.update(inv_index.get(tok, []))
        result.append(cand)
    return result


def generate_candidates(s1_df, other_df, other_source_label, top_k=15, sim_threshold=0.15):
    """
    Generate candidate matches from `other_df` (source2 or source3) for every
    row in `s1_df`, grouped by country.

    Returns: dict {source1_entity_id: set(other_entity_id)}
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

        # 1) TF-IDF nearest neighbors
        topk_results = _tfidf_topk(s1_names, other_names, top_k=top_k)
        for i, s1_id in enumerate(s1_ids):
            for j, sim in topk_results[i]:
                if sim >= sim_threshold:
                    candidates[s1_id].add(other_ids[j])

        # 2) Rare-token overlap safety net
        token_results = _token_block(s1_names, other_names)
        for i, s1_id in enumerate(s1_ids):
            for j in token_results[i]:
                candidates[s1_id].add(other_ids[j])

    # Ensure every s1 entity has an entry (possibly empty)
    for s1_id in s1_df["entity_id"]:
        candidates.setdefault(s1_id, set())

    return candidates, other_source_label