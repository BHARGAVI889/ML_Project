"""
Post-processing to squeeze extra precision out of the scored candidates —
this is usually worth more F_0.5 than another round of feature tuning.

Key idea: in a deduplicated reference source (Source 1), it's rare (though
not guaranteed impossible) for the *same* Source-2/Source-3 record to
genuinely belong to two different Source-1 businesses. When two S1 entities
both want the same candidate, the lower-confidence claim is very likely a
false merge — and false merges are exactly what F_0.5 punishes hardest.
`resolve_conflicts` greedily assigns each contested candidate to whichever
S1 entity scored it highest, dropping the loser(s).

Use --exclusive in predict.py to enable this. Always re-check your own
validation F_0.5 with it on vs. off — if your data genuinely has legitimate
1-to-many cases (e.g. franchises), forcing exclusivity can cost you recall.
Measure, don't assume.
"""
from collections import defaultdict


def resolve_conflicts(scored_rows):
    """
    scored_rows: list of dicts, each with keys
        source1_entity_id, other_entity_id, score
    Returns: same list, minus rows where another S1 entity claimed the same
    other_entity_id with a strictly higher score.
    """
    best_for_other = {}
    for row in scored_rows:
        oid = row["other_entity_id"]
        if oid not in best_for_other or row["score"] > best_for_other[oid]["score"]:
            best_for_other[oid] = row

    kept_keys = {(r["source1_entity_id"], r["other_entity_id"]) for r in best_for_other.values()}
    return [r for r in scored_rows if (r["source1_entity_id"], r["other_entity_id"]) in kept_keys]


def per_country_thresholds(val_df, f_beta_fn, countries, default_threshold, beta=0.5):
    """
    Tune a separate decision threshold per country on the validation set,
    falling back to `default_threshold` for any country not seen in
    validation (this matters here: the test set adds France, which never
    appears in training/validation, so it MUST fall back gracefully).
    """
    thresholds = {}
    for country in countries:
        sub = val_df[val_df["country"].str.lower() == country.lower()]
        if sub.empty:
            continue
        best_t, best_f = default_threshold, -1
        true_groups = (
            sub[sub.label == 1].groupby("source1_entity_id")["other_entity_id"]
            .apply(set).to_dict()
        )
        for s1_id in sub["source1_entity_id"].unique():
            true_groups.setdefault(s1_id, set())
        for t in [i / 100 for i in range(30, 96, 2)]:
            pred_groups = (
                sub[sub.score >= t].groupby("source1_entity_id")["other_entity_id"]
                .apply(set).to_dict()
            )
            f = f_beta_fn(true_groups, pred_groups, beta=beta)
            if f > best_f:
                best_f, best_t = f, t
        thresholds[country.lower()] = best_t
    return thresholds
