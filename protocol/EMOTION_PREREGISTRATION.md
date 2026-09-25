# Preregistration: few-shot Jev on Emotion

This is a new study, written after the completed AG News result and before any
Emotion request. It uses `dair-ai/emotion` at revision
`cab853a1dbdf4c42c2b3ef2173804746df8825fe`: all 2,000 official test rows form
the untouched scoreboard. Labels are sadness, joy, love, anger, fear, and
surprise; their natural imbalance is retained, so macro-F1 is reported beside
accuracy.

The primary comparison is 6-shot (one training example per label) versus
zero-shot. Secondary conditions are 12 and 24 shots. Five deterministic draws
(seeds 0–4) use a fixed canonical label order. The full matrix is 32,000 Jev
requests. The primary interval is the same target-and-draw hierarchical
bootstrap used for AG News. No result will alter this protocol.
