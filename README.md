# Business Entity Resolution — Pipeline

## Approach
Blocking (TF-IDF char n-gram cosine + rare-token overlap, per-country) →
pairwise feature engineering (name/address string similarity) → LightGBM
binary classifier → per-entity threshold tuned to maximize F_0.5 on a
held-out group split.

No external data/APIs are used anywhere — only the provided TSVs.
LightGBM is Microsoft's implementation (MIT license) and is not a
pretrained "model with parameters" in the LLM sense, so it trivially
satisfies the size/license constraint. If you swap in a pretrained
embedding model for blocking (e.g. an MIT/Apache sentence-transformer),
make sure it is loaded from local/cached weights only — no internet
lookups during the run, to stay compliant with the "no external lookup"
rule.

## Setup
```bash
pip install -r requirements.txt --break-system-packages
```

## 1. Train
```bash
cd src
python3 train.py \
    --train-dir ../dataset/train \
    --model-out ../models/matcher.pkl
```
This builds candidate pairs on the training data using the same blocking
logic used at inference time, labels them from `train_ground_truth.tsv`,
trains the classifier, and prints:
- blocking recall (how many true matches survived candidate generation —
  this is your recall ceiling, watch it closely)
- the best decision threshold and its F_0.5 on a held-out validation split

## 2. Predict on the test set
```bash
python3 predict.py \
    --test-dir ../dataset/test \
    --model ../models/matcher.pkl \
    --out-dir ../output
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## 3. Validate before submitting
```bash
python3 ../utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir ../dataset/test
```

## Tuning ideas (biggest recall/precision levers, in order of impact)
1. **Blocking `top_k` and `sim_threshold`** in `blocking.py` — raise `top_k`
   / lower `sim_threshold` if the recall-ceiling printout during training is
   low; this bounds everything downstream.
2. **Country-specific address parsing** — split out PIN/ZIP codes and
   street numbers as dedicated features instead of generic token overlap;
   this is usually the single biggest precision lever for F_0.5.
3. **`max-matches-per-entity`** in `predict.py` — most S1 entities match 0
   or 1 records; capping candidates per entity after scoring reduces false
   merges (precision), which F_0.5 rewards 2x over recall.
4. Add a **blocking-key** feature (e.g. same first-token of normalized
   name) directly into the classifier rather than only using it for
   candidate generation.
