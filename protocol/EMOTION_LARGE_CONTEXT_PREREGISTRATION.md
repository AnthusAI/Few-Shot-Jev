# Preregistration: can many examples steer Emotion classification?

Written after the completed small-context Emotion study and before this study's
first model call. The question is whether **many** correctly labelled examples
in context can improve Jev's ground-truth alignment when zero-shot Emotion
classification is weak.

The untouched `validation` split (2,000 rows) of `dair-ai/emotion`, revision
`cab853a1dbdf4c42c2b3ef2173804746df8825fe`, is the scoreboard. The previously
scored `test` split is never used here. Training examples are sampled only from
`train`, with five nested deterministic draws (seeds 0–4), in canonical class
order: sadness, joy, love, anger, fear, surprise.

The ladder is 0, 1, 4, 16, and 64 examples per label: 0, 6, 24, 96, and 384
total examples. The primary contrast is zero-shot versus 64 examples per
label. **Macro-F1** is the primary metric because Emotion's label distribution
is imbalanced. Accuracy, per-class recall, multiclass Brier score, log loss,
and calibration are supporting measurements. The expected request count is
42,000: 2,000 zero-shot plus four nonzero levels × five draws × 2,000 targets.

The target-and-draw hierarchical bootstrap, 2,000 resamples and seed
`20260928`, supplies the primary 95% interval. Intermediate levels test the
dose-response shape; none may replace the 64-per-label primary contrast after
collection.
