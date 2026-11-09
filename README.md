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

On the same replies it scores behaviour with the benchmark's own judge, which always sees the Original question, and counts
mentions of a test in the model's reasoning.

## Files

| File | What it does |
|---|---|
| `main.py` | the task: `-T benchmark=mask` or `strongreject`, `-T versions=original,cued,control` |
| `cue.py` | the five cues, and the Control's neutral text for each |
| `template.py` | puts the cues around any question and gives the Original, Cued and Control versions |
| `dataset.py` | loads a benchmark's questions and builds their versions |
| `solver.py` | the target model's reply (reasoning effort, reply limit) |
| `scorer.py` | MASK's judge, StrongREJECT's judge, and the count of test mentions in the reasoning |
| `compare.py` | compares the versions: each one against the Original, and Cued against Control |
| `mask/` | MASK's 130 scenarios, its judge prompt, and `extra.py`: the paper's extra versions |
| `strongreject/` | StrongREJECT's 313 prompts, their cue inputs and its judge prompt |

## Setup

Python 3.11 or later, from the repository root:

```bash
pip install inspect_ai==0.3.263 anthropic openai
```

Put the API keys in a `.env` file in the repository root (it is ignored by git): `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`.

## Running

Run from the repository root with `PYTHONPATH=.`. MASK, all three versions in one run:

```bash
inspect eval main.py -T benchmark=mask --model openrouter/nvidia/nemotron-3.5-super-vl:free --sample-shuffle 42 --max-connections 4 --log-dir logs-mask
```

StrongREJECT, one version per run as in the paper (`original`, `cued` or `control`):

```bash
inspect eval main.py -T benchmark=strongreject -T versions=cued --model anthropic/claude-opus-5-5 -T effort=medium --sample-shuffle 42 --log-dir logs-sr
```

Compare the versions:

```bash
python compare.py logs-mask
```

The judge is Claude Sonnet 5.5 by default (`-T judge=...` changes it). The paper's settings: Claude targets with
`-T effort=medium`, and on MASK `-T max_tokens=3500`. The extra MASK versions run the same way, for example
`-T versions=original,cues_1_4,honesty_request`.

## Adding a benchmark

A benchmark needs its questions, its own judge (shown the Original question), and the two words of its grading line, which
`template.fill_grading()` writes once from the benchmark's description. `template.cue_inputs()` picks each question's names
and sentence, and `template.versions()` builds the three versions.

## Licence

Code: MIT (`LICENSE`). The cue kinds are adapted from EvalAwareBench (Li et al. 2026, arXiv:2605.23055). MASK and StrongREJECT
keep their own licences.
