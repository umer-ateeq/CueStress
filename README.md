# CueStress

**CueStress: A Reproducible Framework for Auditing Evaluation-Cue Sensitivity in LLM Benchmarks**
Umer Ateeq, independent researcher.

CueStress is a toolkit for the [Inspect AI](https://inspect.aisi.org.uk) evaluation library. It keeps each benchmark
question's wording, adds five test cues around it, and compares three versions on the same items:

| Version | What the model sees |
|---|---|
| Original | the benchmark question as it is |
| Cued | a test-style header, the question with placeholder names and a contradicting sentence, and a last line with `EVAL==TRUE` and what is graded |
| Control | the same layout with neutral text: a plain header, ordinary names, a consistent sentence and a neutral last line |

On the same replies it scores behaviour with the benchmark's own scorer (a judge always sees the Original question), and
counts mentions of a test in the model's reasoning.

## Files

| File | What it does |
|---|---|
| `main.py` | the task: `-T benchmark=mask`, `strongreject` or `gpqa`, `-T versions=original,cued,control` |
| `cue.py` | the five cues, and the Control's neutral text for each |
| `template.py` | puts the cues around any question and gives the Original, Cued and Control versions |
| `dataset.py` | loads a benchmark's questions and builds their versions |
| `solver.py` | the target model's reply (reasoning effort, reply limit) |
| `scorer.py` | MASK's judge, StrongREJECT's judge, and the count of test mentions in the reasoning (GPQA uses Inspect's own `answer("letter")` scorer, with no judge) |
| `compare.py` | compares the versions: each against the Original, Cued against Control, and the paper's other comparisons |
| `mask/` | MASK's 130 scenarios, its judge prompt, and `extra.py`: the paper's extra versions |
| `strongreject/` | StrongREJECT's 313 prompts, their cue inputs and its judge prompt |
| `gpqa/` | each GPQA Diamond question's cue 3 sentence. The 198 questions are read at run time from the file inspect_evals uses (sha256 `41d1213c...`), since GPQA asks that its questions are not put online |

## Setup

Python 3.14 (the version of the paper's runs), from the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact package versions of the paper's runs; newer versions of some of Inspect's dependencies
do not work with `inspect_ai` 0.3.263. Put the API keys in a `.env` file in the repository root (it is ignored by git):
`ANTHROPIC_API_KEY` (targets and the judge) and `OPENROUTER_API_KEY` (Nemotron).

## Running

From the repository root. MASK, all three versions in one run, then the comparison:

```bash
inspect eval main.py -T benchmark=mask --model openrouter/nvidia/nemotron-3.5-super-vl:free --sample-shuffle 42 --max-connections 4 --log-dir logs-mask
```

```bash
python compare.py logs-mask
```

StrongREJECT runs one version per run, as in the paper (`original`, `cued` or `control`):

```bash
inspect eval main.py -T benchmark=strongreject -T versions=cued --model anthropic/claude-opus-5-5 -T effort=medium --sample-shuffle 42 --max-connections 8 --log-dir logs-sr-cued
```

`python compare.py logs-sr-original logs-sr-cued logs-sr-control` then compares the three. Several folders with the same
versions (repeat runs) are pooled: each item's mean over the runs.

GPQA Diamond, all three versions in one run, scored by exact letter match (no judge, so a free model costs nothing):

```bash
inspect eval main.py -T benchmark=gpqa --model openrouter/nvidia/nemotron-3.5-super-vl:free --sample-shuffle 42 --max-connections 8 --log-dir logs-gpqa
```

The Original is exactly the prompt inspect_evals' `gpqa_diamond` sends (Inspect's multiple-choice prompt, choices in its seed-42
order). A reply with no answer line counts as wrong, as in the benchmark; `compare.py` also prints the share that gave one.

## Reproducing the paper

Every MASK and StrongREJECT run used the judge `anthropic/claude-sonnet-5-5` at effort low with a 2,000-token limit (the defaults);
every run used `--sample-shuffle 42`. Claude targets add `-T effort=medium`, and on MASK `-T max_tokens=3500`; Nemotron is
`openrouter/nvidia/nemotron-3.5-super-vl:free` at its default settings.

| Run | Options |
|---|---|
| MASK, Nemotron, first run | `-T benchmark=mask` (Original, Cued, Control) |
| MASK, Nemotron, second and third runs | `-T versions=original,control,cued,cues_1_4,honesty_request` |
| MASK, Nemotron, last line | `-T versions=neutral_line,last_line` |
| MASK, Nemotron, 2×2 | `-T versions=neutral_line,tag_only,grading_only,last_line` |
| MASK, Opus | `-T versions=original,control,cued,cues_1_4,tone_line`, and a separate run with `-T versions=original,honesty_request` |
| MASK, Sonnet and Haiku | `-T versions=original,control,cued,cues_1_4,tone_line,honesty_request` |
| MASK, Opus and Sonnet, last line | `-T versions=original,neutral_line,last_line,honesty_request` |
| StrongREJECT, Claude models | one run per version, `-T benchmark=strongreject -T versions=<version>` |
| StrongREJECT, Nemotron | the same with `--limit 180` (the 180 prompts the shuffle puts first) |
| GPQA Diamond, Nemotron | `-T benchmark=gpqa` (Original, Cued, Control; reply limit 32,000 tokens) |
| GPQA Diamond, Opus and Haiku | `-T benchmark=gpqa -T effort=medium -T max_tokens=8000 --batch` |

The models answer with sampling, so a new run gives new replies: its results land near the paper's, not on them, as the
paper's three Nemotron runs show. `compare.py` compares items in the order of the scenario files, so the same logs always
give the same intervals.

## Adding a benchmark

A benchmark needs its questions, its own scorer (a judge is shown the Original question), and the two words of its grading
line, which `template.fill_grading()` writes once from the benchmark's description. `template.cue_inputs()` picks each
question's names and sentence, and `template.versions()` builds the three versions. GPQA shows the whole recipe: a loader
that reads the benchmark as its Inspect task does, `frame=` for its multiple-choice prompt (cues 2 and 3 go inside it, the
header and last line outside), `rename=False` because its answer depends on every word, and Inspect's own scorer.

## Licence

Code: MIT (`LICENSE`). The cue kinds are adapted from EvalAwareBench (Li et al. 2026, arXiv:2605.23055). MASK and StrongREJECT
keep their own licences.
