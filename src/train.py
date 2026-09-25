"""
Train the pairwise matching classifier.

Usage:
    python3 train.py \
        --train-dir dataset/train \
        --model-out models/matcher.txt

Builds candidates on the training data with the same blocking function used
at inference time, labels each candidate pair using train_ground_truth.tsv,
trains a LightGBM binary classifier, and reports held-out F_0.5.
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from blocking import generate_candidates
from data_utils import load_source, load_ground_truth
from features import pair_features, FEATURE_COLUMNS
from postprocess import per_country_thresholds


def build_labeled_pairs(train_dir: Path):
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")

    gt_map = {}
    for _, row in gt.iterrows():
        matched = row["matched_entity_ids"].strip()
        gt_map[row["source1_entity_id"]] = (
            set(m.strip() for m in matched.split(",")) if matched else set()
        )

    s2_lookup = s2.set_index("entity_id").to_dict("index")
    s3_lookup = s3.set_index("entity_id").to_dict("index")
    s1_lookup = s1.set_index("entity_id").to_dict("index")

    cand2, _ = generate_candidates(s1, s2, "S2")
    cand3, _ = generate_candidates(s1, s3, "S3")

    rows = []
    for s1_id, s1_row in s1_lookup.items():
        true_matches = gt_map.get(s1_id, set())
        for other_id in cand2.get(s1_id, set()):
            feats = pair_features(s1_row, s2_lookup[other_id])
            feats.update(source1_entity_id=s1_id, other_entity_id=other_id,
                         label=int(other_id in true_matches))
            rows.append(feats)
        for other_id in cand3.get(s1_id, set()):
            feats = pair_features(s1_row, s3_lookup[other_id])
            feats.update(source1_entity_id=s1_id, other_entity_id=other_id,
                         label=int(other_id in true_matches))
            rows.append(feats)

    df = pd.DataFrame(rows)
    # Recall check: how many ground-truth matches survived blocking?
    all_true = sum(len(v) for v in gt_map.values())
    recovered = df[df.label == 1]["source1_entity_id"].nunique()
    print(f"[blocking] total S1 entities with >=1 true match: "
          f"{sum(1 for v in gt_map.values() if v)}, "
          f"recovered at least one match for: {recovered}")
    print(f"[blocking] total candidate pairs: {len(df)}, positive pairs: {df.label.sum()} "
          f"(true match links: {all_true})")
    return df


def f_beta_per_entity(y_true_groups, y_pred_groups, beta=0.5):
    """y_true_groups / y_pred_groups: dict s1_id -> set(matched other ids)."""
    scores = []
    for s1_id, true_set in y_true_groups.items():
        pred_set = y_pred_groups.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        scores.append(f)
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--model-out", type=Path, required=True)
    ap.add_argument("--threshold-search", action="store_true", default=True)
    args = ap.parse_args()

    df = build_labeled_pairs(args.train_dir)

    # Group-aware split so no Source-1 entity leaks across train/val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["source1_entity_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

    # class imbalance: weight positives up
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    val_df = val_df.copy()
    val_df["score"] = model.predict_proba(X_val)[:, 1]

    # Threshold sweep to maximize F_0.5 (precision-heavy)
    y_true_groups = (
        val_df[val_df.label == 1].groupby("source1_entity_id")["other_entity_id"]
        .apply(set).to_dict()
    )
    for s1_id in val_df["source1_entity_id"].unique():
        y_true_groups.setdefault(s1_id, set())

    best_t, best_f = 0.5, -1
    for t in [i / 100 for i in range(30, 96, 2)]:
        pred_groups = (
            val_df[val_df.score >= t].groupby("source1_entity_id")["other_entity_id"]
            .apply(set).to_dict()
        )
        f = f_beta_per_entity(y_true_groups, pred_groups, beta=0.5)
        if f > best_f:
            best_f, best_t = f, t

    print(f"[val] best global threshold={best_t:.2f}  F_0.5={best_f:.4f}")

    # Per-country thresholds (test set adds France, unseen here — predict.py
    # falls back to the global threshold for any country not in this dict)
    val_df_with_country = val_df.merge(
        df.drop_duplicates("source1_entity_id")[["source1_entity_id"]],
        on="source1_entity_id", how="left",
    )
    # attach country from the original s1 frame for threshold tuning
    s1_countries = load_source(args.train_dir / "train_source1.tsv")[["entity_id", "country"]]
    val_df_with_country = val_df.merge(
        s1_countries, left_on="source1_entity_id", right_on="entity_id", how="left"
    )
    country_thresholds = per_country_thresholds(
        val_df_with_country, f_beta_per_entity,
        countries=val_df_with_country["country"].dropna().unique().tolist(),
        default_threshold=best_t,
    )
    print(f"[val] per-country thresholds: {country_thresholds}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "threshold": best_t,
            "country_thresholds": country_thresholds,
            "features": FEATURE_COLUMNS,
        }, f)
    print(f"Saved model + thresholds to {args.model_out}")


if __name__ == "__main__":
    main()
