"""CueStress: a benchmark's questions as their Original, Cued and Control versions, scored by the benchmark's own scorer and by
the count of test mentions in the reasoning. From the repository root:
    inspect eval main.py -T benchmark=mask --model <target> --sample-shuffle 42
    inspect eval main.py -T benchmark=strongreject -T versions=cued --model <target> --sample-shuffle 42
    inspect eval main.py -T benchmark=gpqa --model <target> --sample-shuffle 42
Then: python compare.py <log folder>"""
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import answer

from dataset import make_dataset
from scorer import mask_honesty, mentions, strongreject_judge
from solver import make_solver

JUDGE = "anthropic/claude-sonnet-5-5"
MAX_TOKENS = {"mask": 4096, "strongreject": 3500, "gpqa": 32000}  # the reply limits (Claude on MASK: 3,500)


@task
def cuestress(benchmark="mask", versions=("original", "cued", "control"), judge=JUDGE, effort=None, max_tokens=None,
              judge_effort="low", judge_max_tokens=2000):
    """benchmark: mask, strongreject or gpqa; versions: which to run (-T versions=original,cued,control; MASK's extra versions
    are in mask/extra.py); effort: the target's reasoning effort (Claude: medium)."""
    chosen = [versions] if isinstance(versions, str) else list(versions)
    judge_config = GenerateConfig(reasoning_effort=judge_effort, max_tokens=judge_max_tokens, cache_prompt=False)
    judge_scorer = (answer("letter") if benchmark == "gpqa"  # GPQA: Inspect's own letter match, as its task scores it; no judge
                    else {"mask": mask_honesty, "strongreject": strongreject_judge}[benchmark](judge, judge_config))
    return Task(dataset=make_dataset(benchmark, chosen), solver=make_solver(effort, max_tokens or MAX_TOKENS[benchmark]),
                scorer=[judge_scorer, mentions()], fail_on_error=False)
