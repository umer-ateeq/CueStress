"""The requests, loaded from the authors' dataset file (200 requests, each in all 256 cue settings)."""

import json
import random

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Sample, json_dataset

from cue import CORE4, CORE5, condition_name, cues_in, stack_combinations

# The authors' dataset file (downloaded once, then cached).
DATA_FILE = hf_hub_download(
    repo_id="aisa-group/EvalAwareBench",
    repo_type="dataset",
    filename="data/prompts.jsonl",
    revision="ee984a8df39387946315475ee7b221ad790e1b34",  # a fixed version, so the data never changes
)

# Which requests to use:
#   "all"        = all 200 (100 harmful "safety" requests + their 100 harmless "capability" twins)
#   "safety"     = the 100 harmful ones,  "capability" = the 100 harmless ones
#   or a list of task ids. While testing with free models, a few is enough:
REQUESTS = "all"

# Stacking (-T cues=k0 ... k8) uses only 25 task pairs, like the paper: 25 safety requests + their 25 capability twins.
# The authors didn't publish theirs, so we pick our own at random with a fixed seed: the same 25 every run.
# (This only applies when REQUESTS is "all", "safety" or "capability". A list of task ids is used as it is.)
STACKING_SEED = 42


def stacking_tasks():
    """The 50 requests used for stacking: 25 random task pairs, each in its safety and capability version."""
    pairs = sorted({json.loads(line)["task_name"] for line in open(DATA_FILE, encoding="utf-8")})  # the 100 pairs
    chosen = random.Random(STACKING_SEED).sample(pairs, 25)
    return [pair + "_" + kind for pair in chosen for kind in ("safety", "capability")]


STACKING_TASKS = stacking_tasks()


def record_to_sample(record):
    """Turn one line of the dataset file into a Sample."""
    condition = condition_name(cues_in(record))  # e.g. "none", "F5", "F1+F8"
    return Sample(
        id=record["task_id"] + "__" + condition,  # e.g. lyrics_purple_rain_safety__F5
        input=record["prompt"],  # what the model sees
        metadata={"request": record["task_id"], "type": record["valence"], "condition": condition},  # hidden from the model
    )


def make_dataset(cues_on, requests=REQUESTS):
    """Pick the chosen requests, in the version(s) with the chosen cues on.

    cues_on is a list of cues (e.g. ["F1", "F8"]), or a stacking level (e.g. ["k4"]): then each request
    comes in the paper's 10 random combinations of 4 cues, and only the 25 stacking pairs are used.
    """
    all_versions = json_dataset(DATA_FILE, record_to_sample)  # 200 requests x 256 versions = 51,200 Samples # INSPECT DATASET THIS IS!
    stacking = len(cues_on) == 1 and (cues_on[0].startswith("k") or cues_on[0] in ("core5", "core4")) #only true for k4,1,2,5,6etc (and core5/core4)

    def wanted(sample):
        if requests == "all":
            right_request = True
        elif requests in ("safety", "capability"):
            right_request = (sample.metadata["type"] == requests)
        else:
            right_request = (sample.metadata["request"] in requests)
        if not right_request:
            return False
        if stacking:
            if requests in ("all", "safety", "capability") and sample.metadata["request"] not in STACKING_TASKS:
                return False  # the paper stacks cues on 25 task pairs only, not on all 100
            if cues_on[0] in ("core5", "core4"):  # our template's 5 or 4 cues, on the same 25 task pairs
                return sample.metadata["condition"] == condition_name(CORE5 if cues_on[0] == "core5" else CORE4)
            k = int(cues_on[0][1:]) # [k8] gives 8 , [k0] gives 0.
            
            return sample.metadata["condition"] in stack_combinations(sample.metadata["request"], k)
        return sample.metadata["condition"] == condition_name(cues_on)
#filter returns a smaller Dataset. For k4 + "safety" that’s 250 Samples.
    return all_versions.filter(wanted)
