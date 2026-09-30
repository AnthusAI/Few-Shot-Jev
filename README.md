# Does Jev learn a classification boundary from four examples?

> A model can recognize a familiar category without being told what a team means by it. A labeled example may make that boundary clearer—or it may add noise, a misleading prototype, or an accidental shortcut.

Jev is often used as a zero-shot decision model: provide a piece of text, name the possible outcomes, and receive a typed choice with probabilities. But its state can also contain related labeled records. That makes a simple and useful question testable: **when Jev is given a few examples of a classification task, does it agree with the task's ground truth more often?**

This repository is a deliberately small, preregistered experiment on AG News. It is not a general benchmark. The article below explains the hypothesis, the measurement, and what the completed frozen run does—and does not—show.

## The hypothesis

AG News asks a four-way question about a short news article: is its primary topic **World**, **Sports**, **Business**, or **Sci/Tech**? The label names are fairly readable, so zero-shot classification is a reasonable baseline. Still, the boundaries are not perfectly obvious. A story about a technology company's earnings may sound both Business and Sci/Tech; a sports story about public policy may sound like World.

Our primary hypothesis is modest:

> **One correctly labeled training example for each class will improve Jev's held-out AG News accuracy over the same zero-shot question.**

There are competing possibilities. The examples may not help because the label names already express the task. They may help at four examples but plateau or reverse at eight and sixteen. They may improve the selected label while making reported probabilities less calibrated. Those are reasons to measure a curve, not to assume that more context is always better.

## How we will investigate it

The dataset is [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news), pinned to revision `eb185aade064a813bc0b7f42de02595523103ca4`. We select a fixed, balanced scoreboard of 2,000 articles from its official test split: 500 per class. The checked-in [scoreboard manifest](manifests/scoreboard.jsonl) contains only each upstream row ID, label, and normalized-text hash—not the article text.

All labeled examples come from the official training split. A training article whose normalized text exactly matches a scoreboard article is excluded. No test label influences example selection, prompt writing, or model calls. That separation matters: a few-shot prompt is allowed to teach Jev the task from training labels; it is not allowed to reveal the answer to an item being scored.

Every condition asks one unchanged choice question. The target and examples are separate named fields in Jev's structured state:

```json
{
  "labeled_examples": [
    {"text": "…a labeled training article…", "label": "Sports"}
  ],
  "target": {"text": "…a different, held-out test article…"}
}
```

The question explicitly says to classify only `target.text`; the examples show the intended categories and are not themselves answers to score. In the zero-shot arm, `labeled_examples` is an empty list. The instructions, question type, four criteria, target set, and request shape otherwise stay identical. The actual request builder is small enough to inspect in [`prompt.py`](src/jev_fewshot/prompt.py).

## One example is not a result

Prompt examples are themselves a source of sampling noise. A particularly typical Business article might make the task look easier than another equally valid Business article. To avoid treating one lucky set as the method, the experiment draws **five** reproducible, class-balanced sets of four training examples per class, using seeds 0 through 4.

Each draw creates a nested ladder:

| Condition | Examples shown | Role |
| --- | ---: | --- |
| Zero-shot | 0 | Baseline |
| 4-shot | 1 per class | **Primary comparison** |
| 8-shot | 2 per class | Secondary dose-response result |
| 16-shot | 4 per class | Secondary dose-response result |

The 4-shot set is a prefix of the 8-shot set, which is a prefix of the 16-shot set. In every draw, examples appear in one canonical class order: World, Sports, Business, then Sci/Tech. That makes the ladder answer a clean question: what changes when more examples are added, rather than when an entirely different prompt happens to win?

The complete matrix is 32,000 decisions: 2,000 zero-shot requests plus 2,000 targets × 3 few-shot levels × 5 draws. The runner is resumable and saves only the response, usage, latency, target manifest ID, condition, draw seed, and a hash of the full request. It never saves article text or credentials.

## Could the order of examples be doing the work?

Yes. A model may treat the first example as more salient, favor recently seen categories, or respond differently when the same examples are rearranged. The first run intentionally does **not** try to answer that question. It uses one ordering so it can establish the initial zero-shot versus few-shot result without turning the primary experiment into an unbounded prompt search.

Once that initial Jev result is collected and reported, the next investigation is an ordering-sensitivity study on Jev alone: hold the target set and the selected examples fixed, vary only their presentation order, and compare each reordered result with the canonical order. That follow-up belongs before any comparison with other models. It will be published as a distinct, explicitly labeled analysis—not folded back into the preregistered primary result.

## What counts as evidence

The primary measurement is the **paired held-out accuracy difference: 4-shot minus zero-shot**. Each target is evaluated in both conditions, so the question is not “are two independently sampled scores different?” but “on how many of these exact same articles did the four examples change a correct decision into an incorrect one, or vice versa?”

The reported 95% interval resamples both test targets and the five example draws. We will also report macro-F1, per-class recall, multiclass log loss, multiclass Brier score, top-label calibration error, reliability-bin data, latency, request totals, and returned token usage. Calibration is not a decorative add-on: an arm that raises accuracy by becoming confidently wrong more often deserves a different reading from one that improves both decisions and probabilities.

The [preregistration](protocol/PREREGISTRATION.md) fixes these choices before any live Jev request. It makes 4-shot versus zero-shot primary; 8- and 16-shot results remain useful, but cannot be promoted after the fact just because they look better.

## What we found

We collected all **32,000** preregistered Jev responses with model version `jev-1.13.0`. The zero-shot baseline classified 1,759 of 2,000 held-out articles correctly: **87.95% accuracy**. Across the five fixed 4-shot example draws, accuracy averaged **88.75%**—an apparent gain of **0.80 percentage points**.

That is not enough to call the hypothesis confirmed. The preregistered hierarchical bootstrap interval for the primary 4-shot-minus-zero-shot accuracy difference is **−0.18 to +1.84 percentage points**. It overlaps zero: these five example draws provide some positive movement, but they do not establish a reliable 4-shot improvement on this scoreboard.

| Condition | Held-out accuracy | Mean Brier | Mean ECE |
| --- | ---: | ---: | ---: |
| Zero-shot | 87.95% | .197 | .072 |
| 4-shot, five-draw mean | 88.75% | .183 | .064 |
| 8-shot, five-draw mean | 88.76% | .183 | .064 |
| 16-shot, five-draw mean | 89.04% | .178 | .061 |

The larger-shot rows are secondary analyses, not replacements for the predeclared primary comparison. They are directionally better on average, but also vary materially by example draw: 16-shot accuracy ranges from 87.95% to 90.15%. This is a useful warning against saying simply that “few-shot helps.” Which examples are supplied matters.

The checked-in [aggregate summary](results/summary.json) contains every condition’s accuracy, macro-F1, recall, calibration, latency, and token totals; the per-request cache remains local and contains no article text.

## What we can conclude—and what we cannot

On this fixed AG News evaluation, the selected training examples yielded a small average accuracy improvement, with better average Brier and calibration scores, but the primary interval does not rule out no 4-shot effect. The honest conclusion is therefore **inconclusive rather than positive**.

This result does not prove that few-shot prompting works in general, that examples cause the effect through any particular internal mechanism, or that AG News labels are the right boundary for another team's task. It is a measurement of one model, one request design, and one public classification dataset. Its value is that the claim remains proportionate to the evidence.

## Does a different dataset change the answer?

We then ran the same frozen ladder on
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion): six
emotion labels and its complete, naturally imbalanced 2,000-item official test
split. The primary comparison is 6-shot versus zero-shot—one training example
per emotion label—with five fixed example draws.

Again, the accuracy result is inconclusive. Zero-shot accuracy was **59.35%**;
the 6-shot mean was **59.46%**, a **+0.11-point** change. Its hierarchical 95%
interval was **−1.15 to +1.28 points**. That does not establish an accuracy
gain.

But the supporting measurements moved more clearly: macro-F1 rose from .500 to
.521 and multiclass Brier improved from .660 to .629. At 24 shots, secondary
accuracy reached 60.23%, macro-F1 .539, and Brier .612. The full
[Emotion aggregate](results/emotion_summary.json) records the result.

So the two datasets told a narrower story than “few-shot helps”: few examples
did not produce a reliable primary accuracy gain on either task, but they did
consistently improve probability quality and, on Emotion, class-balanced
performance. That result motivated a different, preregistered question: can
**many** examples move a classifier whose zero-shot alignment is modest?

## Does a lot of context change the boundary?

The small-context experiment was not designed to answer that question. Its
largest dose was only four examples per emotion. We therefore froze a new
[large-context preregistration](protocol/EMOTION_LARGE_CONTEXT_PREREGISTRATION.md)
before making another live call. It uses the same pinned Emotion revision and
the same six-way question, but moves to the untouched 2,000-item official
**validation** split. The earlier test split plays no role in this score.

Each of five nested training draws supplies 0, 1, 4, 16, or 64 examples per
emotion: **0, 6, 24, 96, or 384 total demonstrations**. The examples retain
the same canonical label order in every condition. The primary comparison is
zero-shot versus 384-shot and the primary metric is macro-F1, because the
Emotion labels are imbalanced. This is deliberately a dose-response study, not
an attempt to find a single favorable prompt.

We evaluated all 42,000 planned scoreboard records with `jev-1.13.0`. Forty-
two records had an identical state/question fingerprint to another record, so
the cache made **41,958 distinct model calls** and expanded each cached answer
back to every preselected scoreboard item for analysis. No text is stored in
the published artifact.

| Demonstrations | Accuracy | Macro-F1 | Brier score |
| --- | ---: | ---: | ---: |
| 0 | 58.85% | .515 | .664 |
| 6 (five-draw mean) | 58.72% | .530 | .640 |
| 24 (five-draw mean) | 59.91% | .542 | .618 |
| 96 (five-draw mean) | 62.18% | .577 | .580 |
| 384 (five-draw mean) | 65.74% | .614 | .523 |

The primary 384-shot-versus-zero-shot macro-F1 difference is **+9.89 points**.
Its preregistered target-and-draw hierarchical bootstrap 95% interval is
**+7.62 to +12.26 points**. Accuracy also rose by 6.89 points on the same
scoreboard, and probability quality improved substantially. Unlike the
small-context primary comparisons, this is strong evidence that this large,
fixed set of correctly labelled examples can steer Jev toward the Emotion
ground-truth boundary.

It is not evidence that every large example set works, that the effect is
linear on every task, or that the examples' **order** is irrelevant. At 384
examples, the five fixed draws ranged from .608 to .627 macro-F1. The next
study should therefore hold the target set and selected examples fixed and vary
only presentation order. It belongs before comparing Jev with other models:
otherwise a cross-model result could accidentally be an order effect in
disguise. The [large-context aggregate](results/emotion_large_context_summary.json)
contains the per-draw, pooled, calibration, latency, and token summaries.

## Reproduce the investigation

The tests use only local synthetic data and a fake client; they make no network or model calls.

```bash
make install
make test
make preflight
```

`make preflight` downloads the pinned AG News revision, rebuilds the frozen selection, and enumerates the 32,000 requests without constructing a Jev client or reading an API key. It is the final check before any paid work.

After reviewing that output, place `TYPESAFE_API_KEY` in a gitignored `.env` file and explicitly set the request ceiling you approve:

```bash
MAX_REQUESTS=32000 make run
make report
```

The report writes aggregate metrics to `results/summary.json`; live response rows remain local and ignored by Git. See the frozen [protocol](protocol/PREREGISTRATION.md) for the complete rules.

## License and data

The benchmark harness is [MIT-licensed](LICENSE). AG News remains subject to its upstream terms; this repository does not redistribute its article text.
