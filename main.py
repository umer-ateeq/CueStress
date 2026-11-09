"""Run the evaluation (from this folder):

    inspect eval main.py -T cues=none        # no cues: written like a real user
    inspect eval main.py -T cues=F8          # one cue on
    inspect eval main.py -T cues=F1+F8       # several cues on (join them with +)
    inspect eval main.py -T cues=all         # all 8 cues on

Then `inspect view` to see every request, the model's thinking and the score.
To try another model without editing this file, add e.g. -T model=openrouter/openai/gpt-oss-20b
"""

from inspect_ai import Task, task

from cue import parse_cues
from dataset import make_dataset
from scorer import paper_judge
from solver import make_solver

# Free models while we learn ($0).
MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"            # the model we test (paper: gpt-oss-20b)
JUDGE = "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"  # the model that judges (paper: GPT-5)


@task
def eval_aware(cues="none", model=MODEL, judge=JUDGE):
    """Build the evaluation. cues: "none", "all", or the cues to switch on, e.g. "F8" or "F1+F8"."""
    cues_on = parse_cues(cues)  # e.g. "F1+F8" -> ["F1", "F8"]
    return Task(
        dataset=make_dataset(cues_on),
        solver=make_solver(),
        scorer=paper_judge(judge),
        model=model,           # which model to test
        epochs=3,              # ask each request 3 times, like the paper
        fail_on_error=False,   # if one try breaks, skip it and keep going
    )
