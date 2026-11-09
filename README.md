# CueStress

**When a Test Cue Is Also an Instruction: Separating Evaluation Signals from Grading Instructions**
Umer Ateeq, independent researcher.

CueStress is a toolkit for the [Inspect AI](https://inspect.aisi.org.uk) evaluation library. It keeps each benchmark
question's wording, adds five test cues around it, and compares three versions on the same items:

| Version | What the model sees |
|---|---|
| Original | the benchmark question as it is |
| Cued | a test-style header, the question with placeholder names and a contradicting sentence, and a last line with `EVAL==TRUE` and what is graded |
| Control | the same layout with neutral text: a plain header, ordinary names, a consistent sentence and a neutral last line |

The benchmark's own judge always sees the Original question, so it scores the reply, not the cues.

## Files

| File | What it does |
|---|---|
| `cue.py` | the five cues, and the Control's neutral text for each |
| `template.py` | puts the cues around any question: `versions()` gives the Original, Cued and Control versions |
| `mask/task.py` | MASK (honesty under pressure) with the three versions; `mask/scenarios.jsonl` holds its 130 scenarios |
| `mask/extra.py` | the paper's extra MASK versions, which split the grading line from the other cues |
| `strongreject/task.py` | StrongREJECT (refusal of harmful requests) with the three versions |
| `compare.py` | compares the versions in the logs: each one against the Original, and Cued against Control |

## Setup

Python 3.11 or later, from the repository root:

```bash
pip install inspect_ai==0.3.263 anthropic openai python-dotenv
```

Put the API keys in a `.env` file in the repository root (it is ignored by git): `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`.

## Running

Run from the repository root with `PYTHONPATH=.`. MASK, all three versions in one run:

```bash
inspect eval mask/task.py --model openrouter/nvidia/nemotron-3.5-super-vl:free --sample-shuffle 42 --max-connections 4 --log-dir logs-mask
```

StrongREJECT, one version per run (`original`, `cued` or `control`):

```bash
inspect eval strongreject/task.py -T version=cued --model anthropic/claude-opus-5-5 -T effort=medium --sample-shuffle 42 --log-dir logs-sr-cued
```

Compare the versions:

```bash
python compare.py logs-mask
```

The extra MASK versions run the same way, for example `-T versions=original,cues_1_4,honesty_request`.

## Adding a benchmark

A benchmark needs its questions, its own judge (shown the Original question), and the two words of its grading line, which
`template.fill_grading()` writes once from the benchmark's description. `template.cue_inputs()` picks each question's names
and sentence, and `template.versions()` builds the three versions.

## Licence

Code: MIT (`LICENSE`). The cue kinds are adapted from EvalAwareBench (Li et al. 2026, arXiv:2605.23055). MASK and StrongREJECT
keep their own licences.
