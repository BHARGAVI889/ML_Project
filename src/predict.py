"""
Run inference on the test set and write output/matching_results.tsv and
output/candidate_pairs.tsv.

Usage:
    python3 predict.py \
        --test-dir dataset/test \
        --model models/matcher.txt \
        --out-dir output
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from blocking import generate_candidates
from data_utils import load_source
from features import pair_features, FEATURE_COLUMNS
from postprocess import resolve_conflicts


def write_id_list_tsv(path: Path, rows: dict, id_col: str, list_col: str):
    with open(path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in rows.items():
            f.write(f"{s1_id}\t{','.join(sorted(ids))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-matches-per-entity", type=int, default=10)
    ap.add_argument("--exclusive", action="store_true",
                     help="Enforce that each S2/S3 record is claimed by at most one S1 entity "
                          "(keeps the highest-scoring claim). Validate this helps on your own "
                          "held-out split before relying on it — see postprocess.py.")
    args = ap.parse_args()

    s1 = load_source(args.test_dir / "test_source1.tsv")
    s2 = load_source(args.test_dir / "test_source2.tsv")
    s3 = load_source(args.test_dir / "test_source3.tsv")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    default_threshold = bundle["threshold"]
    country_thresholds = bundle.get("country_thresholds", {})

    s1_lookup = s1.set_index("entity_id").to_dict("index")
    s2_lookup = s2.set_index("entity_id").to_dict("index")
    s3_lookup = s3.set_index("entity_id").to_dict("index")

    cand2, _ = generate_candidates(s1, s2, "S2")
    cand3, _ = generate_candidates(s1, s3, "S3")

    candidate_pairs = {}
    all_scored_rows = []  # for optional global exclusivity resolution

    for s1_id, s1_row in s1_lookup.items():
        cand_ids = sorted(cand2.get(s1_id, set()) | cand3.get(s1_id, set()))
        candidate_pairs[s1_id] = set(cand_ids)
        if not cand_ids:
            continue

        rows = []
        for other_id in cand_ids:
            other_row = s2_lookup[other_id] if other_id.startswith("S2-") else s3_lookup[other_id]
            feats = pair_features(s1_row, other_row)
            feats["other_entity_id"] = other_id
            rows.append(feats)

        feat_df = pd.DataFrame(rows)
        feat_df["score"] = model.predict_proba(feat_df[FEATURE_COLUMNS])[:, 1]
        country = str(s1_row.get("country", "")).strip().lower()
        threshold = country_thresholds.get(country, default_threshold)

        passed = feat_df[feat_df.score >= threshold]
        for _, r in passed.iterrows():
            all_scored_rows.append({
                "source1_entity_id": s1_id,
                "other_entity_id": r["other_entity_id"],
                "score": r["score"],
            })

    if args.exclusive:
        all_scored_rows = resolve_conflicts(all_scored_rows)

    # cap matches per S1 entity, keeping the highest-scoring ones
    per_entity = {}
    for row in all_scored_rows:
        per_entity.setdefault(row["source1_entity_id"], []).append(row)

    matching_results = {}
    for s1_id in s1_lookup:
        rows = sorted(per_entity.get(s1_id, []), key=lambda r: -r["score"])
        matching_results[s1_id] = {r["other_entity_id"] for r in rows[: args.max_matches_per_entity]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(args.out_dir / "candidate_pairs.tsv", candidate_pairs,
                       "source1_entity_id", "candidate_entity_ids")
    write_id_list_tsv(args.out_dir / "matching_results.tsv", matching_results,
                       "source1_entity_id", "matched_entity_ids")
    print(f"Wrote {len(matching_results)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
