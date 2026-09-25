# Winning Strategy — Amazon ML Challenge 2026 (Entity Resolution)

## Where teams actually lose points
1. **Blocking recall ceiling.** If your blocking misses a true match, no
   classifier downstream can find it. Most mid-table teams under-invest
   here and cap out around 70-80% recall without noticing, because they
   only look at their final F_0.5, not the recall-ceiling printout.
   `train.py` in the pipeline prints this every run — treat it as your
   #1 metric to watch before you touch the model.
2. **Over-matching on ambiguous pairs.** F_0.5 weights precision 2x over
   recall. A model tuned for F1 or accuracy will systematically over-match
   and lose points it didn't need to lose. Always tune the threshold on
   F_0.5 directly (the pipeline does this), never on a generic metric.
3. **Singletons treated as an afterthought.** A correctly-predicted
   singleton is worth a full 1.0; a false merge on one is worth 0.0. If a
   meaningful fraction of your Source-1 entities are singletons (check this
   in your training data), getting the "no match" case right is often
   worth more marginal F_0.5 than squeezing precision on multi-match
   entities. Don't just optimize the pipeline for matched pairs.
4. **France in the test set.** It doesn't appear in training. Any
   feature or threshold that implicitly hard-codes US/India patterns
   will degrade on it. The pipeline's `country_match` feature and
   per-country threshold fallback are built to handle an unseen country
   gracefully — but review your own address-parsing logic (if you add
   any) for the same trap.

## Highest-leverage upgrades, roughly in order of expected payoff
- **Widen blocking, then trust the classifier to clean up.** It's cheaper
  to raise `top_k` in `blocking.py` and let the classifier reject bad
  candidates than to under-block and never see the true match at all.
- **Country-specific address parsing.** Splitting PIN/ZIP codes and house
  numbers into their own comparison features (instead of generic token
  overlap) is usually the single biggest precision lever, since exact
  numeric matches on PIN codes are very strong match signals.
- **Exclusivity post-processing** (`postprocess.py`, `--exclusive` flag):
  if two different Source-1 entities both claim the same Source-2 record,
  keep only the higher-confidence claim. Cheap to add, meaningfully
  reduces false merges. Validate it actually helps on your own split
  first — don't ship it blind.
- **Per-country thresholds** instead of one global cutoff — different
  countries have different noise profiles, so one threshold is rarely
  optimal for both. Already wired into `train.py`/`predict.py`.
- **Ensembling two model families** (e.g. LightGBM + logistic regression
  on the same features, averaged) is a cheap, low-risk precision bump if
  you have spare time near the end — don't do this before blocking/
  threshold work is solid.

## 72-hour time budget
Use the step plan below. Build in checkpoints for validation, not just
coding — a team that stops to validate F_0.5 locally every few hours will
beat a team that codes for 60 hours straight and submits once.

## Submission discipline
- **5 submissions/day max** — don't burn them on minor tweaks. Validate
  locally with `utils/validate_submission.py` and your own held-out
  F_0.5 first; only submit to the leaderboard when you expect a real
  improvement.
- **Keep every submission's code + score.** The final package review
  checks reproducibility — if your best leaderboard score came from a
  config you didn't save, you can't reproduce it for the audit.
- **Write the methodology doc as you go**, not at the end. You will
  forget why you made a given tuning decision under deadline pressure on
  day 3.

## Documentation that actually scores well
Amazon's reviewers are grading methodology, not just the leaderboard
number for the top 100/finale stages. Your 1-2 page doc should clearly
state, in this order: blocking strategy and its measured recall ceiling,
feature list and why each one helps for this domain (name/address
noise types), model choice and why it satisfies the license/size
constraint, threshold-selection method (and that it targets F_0.5
specifically, not F1/accuracy), and any precision-focused
post-processing (like exclusivity). Reviewers can tell in one page
whether a team understood the metric or just fit a model — make it
obvious you did the former.
