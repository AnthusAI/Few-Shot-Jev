# Preregistration: fixed-budget few-shot selection on Emotion

Written before this study's first Jev call. This tests whether choosing a
96-example few-shot context improves held-out ground-truth alignment beyond
ordinary random prompt-set variation. It does **not** test whether selection is
more important than context size.

The original Emotion `test` and `validation` splits have already been used by
earlier studies and are not used here. We partition the pinned revision
`cab853a1dbdf4c42c2b3ef2173804746df8825fe`'s original `train` split by label,
with seed `20260930`: 100 records per label form a selector-development set;
the frozen 2,000-record scoreboard has label counts sadness 583, joy 670, love
163, anger 270, fear 242, surprise 72; all remaining rows form the candidate
pool. The committed scoreboard manifest records only ID, label, and text hash.

Every few-shot condition has exactly 96 examples: 16 per label, same question,
same canonical label order, and the same target field. The zero-shot baseline
has no examples. The primary score is macro-F1 on the frozen scoreboard.

The selectors are: five seed-fixed random class-balanced contexts; a fixed
class-prototype selector based only on candidate-pool token frequencies; and
per-label lexical nearest-neighbor retrieval using target text and candidate
text, but never target labels. The five random contexts are also evaluated on
the labeled selector-development partition. The single context with highest
development macro-F1 is named the development-selected global context before
the scoreboard is read. Its scoreboard score is compared with the distribution
of all five random contexts, not promoted as a new primary endpoint.

The preflight matrix is 19,000 requests: 3,000 development decisions (five
contexts × 600 targets) and 16,000 scoreboard decisions (zero-shot, five
random contexts, prototype, and retrieval; each × 2,000 targets). Macro-F1,
accuracy, Brier, per-class recall, usage, and latency are reported. A paired
target bootstrap gives intervals for each deterministic selector versus the
mean random context. Selection-run variation is shown directly via the five
random contexts.
