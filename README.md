# CueStress

**When a Test Cue Is Also an Instruction: Separating Evaluation Signals from Grading Instructions**
Umer Ateeq, independent researcher.

CueStress is a toolkit for the [Inspect AI](https://inspect.aisi.org.uk) evaluation library. It keeps each benchmark
question's wording, adds five test cues adapted from EvalAwareBench, and measures, on the same replies, whether the model
notices them (mentions of a test in its reasoning) and whether its behaviour changes.

The five cues are a test-style header, fake names, a self-contradicting sentence, the tag `EVAL==TRUE`, and a grading line
naming what is scored. Every question runs as three versions:

| Version | `arms` name |
|---|---|
| Original | `plain` |
| Cued (cues 1–5) | `allcues` |
| Control (same layout, neutral text) | `placebo` |

The benchmark's own judge always sees the Original question, so it scores the reply, not the cues.

## Layout

| Path | What it holds |
|---|---|
| `cue.py`, `dataset.py` | EvalAwareBench's cues and data (CC BY-NC 4.0 data) |
| `main.py`, `solver.py`, `scorer.py`, `judge_prompt.txt` | the EvalAwareBench task in Inspect: its solver and its awareness judge |
| `template/` | builds the cue versions for a benchmark |
| `mask/` | the MASK task with every version, its scenario files and its build script |
| `strongreject/` | the StrongREJECT task with the Original, Cued and Control versions |

## Setup

Python 3.11 or later, from the repository root:

```bash
pip install inspect_ai==0.3.263 anthropic openai python-dotenv huggingface_hub pyyaml matplotlib
```

Put the API keys in a `.env` file in the repository root (it is ignored by git): `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`
and `OPENROUTER_API_KEY_4`.

## Running

Run every command from the repository root with `PYTHONPATH=.`. MASK, all three versions:

```bash
inspect eval mask/task.py -T condition=mixed -T awareness=false -T version=final6 -T arms=plain,allcues,placebo -T model=openrouter/nvidia/nemotron-3.5-super-vl:free -T judge=anthropic/claude-sonnet-5-5 -T judge_effort=low -T judge_max_tokens=2000 --max-connections 4 --sample-shuffle 42
```

StrongREJECT, one version on one model (`-T version=original`, `cued` or `placebo`):

```bash
inspect eval strongreject/task.py -T version=cued --model anthropic/claude-opus-5-5 -T effort=medium -T judge=anthropic/claude-sonnet-5-5 -T judge_effort=low -T judge_max_tokens=2000 --max-connections 8 --sample-shuffle 42
```

## Licence

Code: MIT (`LICENSE`). EvalAwareBench data: CC BY-NC 4.0, no training use. MASK and StrongREJECT keep their own licences.
