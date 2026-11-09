"""Launch each stage of mask/final_plan.md with exactly the planned flags (so none can be forgotten or mistyped).
Run from simple/:  python <this file> <stage> [<log>] [--spent X] [--go]
stages: check1 / check2 / check3 (the free check, rows 8-10 / 11-13 / 14-16, one OpenRouter key each),
        check4 / check5 / check6 (stage 1b, the free retest, rows 17-19 / 20-22 / 23-25),
        smoke --spent X (paid, 10 samples), full --spent X (paid, 650 samples, once only),
        retry <log> --spent X (paid: one eval-retry of the errored and missing samples, only if they fit the money left).
The paid stages run all 5 arms with only the honesty judge (amendment 7; amendment 6 had cut them to plain and allcues);
the free stages run all 5 arms and both judges, as before.
X = the paid spend so far: the larger of the Anthropic console figure (read about 10 minutes after the last call)
and the sum of the computed costs of all paid logs so far.
Without --go it only prints the command. Paid stages also need the owner's go-ahead for that stage."""
import os
import subprocess
import sys

from dotenv import dotenv_values
from inspect_ai.log import read_eval_log

ENV = dotenv_values(".env")
INSPECT = r"..\..\.venv\Scripts\inspect"
BASE = ["eval", "mask/task.py", "-T", "condition=mixed", "-T", "awareness=true", "--sample-shuffle", "42"]
PAID_BASE = ["eval", "mask/task.py", "-T", "condition=mixed", "-T", "awareness=false", "-T", "arms=plain,placebo,allcues,allcues_wrong,allcues_flag", "--sample-shuffle", "42"]
PAID = ["-T", "model=anthropic/claude-opus-5-5", "-T", "effort=medium", "-T", "max_tokens=3500", "-T", "judge=anthropic/claude-sonnet-5-5",
        "-T", "judge_effort=low", "-T", "judge_max_tokens=2000", "--max-connections", "8"]
FREE = {"check1": ("OPENROUTER_API_KEY", 8, 11), "check2": ("OPENROUTER_API_KEY_2", 11, 14), "check3": ("OPENROUTER_API_KEY_3", 14, 17),
        "check4": ("OPENROUTER_API_KEY", 17, 20), "check5": ("OPENROUTER_API_KEY_2", 20, 23), "check6": ("OPENROUTER_API_KEY_3", 23, 26)}  # 4-6: stage 1b
# the hard ceiling of the most expensive paid sample (any of the 5 arms, honesty judge only, 2 chars/token) and the full run's size
BUDGET, CEILING_MAX, FULL_N = 96.0, 0.111, 650

argv = sys.argv[1:]
spent = float(argv[argv.index("--spent") + 1]) if "--spent" in argv else None
words = [a for i, a in enumerate(argv) if a != "--go" and a != "--spent" and not (i and argv[i - 1] == "--spent")]
stage = words[0] if words else None
assert stage in (*FREE, "smoke", "full", "retry"), f"stage must be one of {(*FREE, 'smoke', 'full', 'retry')}, not {stage}"
# a clean environment: only the planned key, and no VS Code variables (in VS Code, Inspect reloads .env over the key set here)
env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "TERM_PROGRAM")
       and not k.startswith("VSCODE")}
env["PYTHONPATH"] = "."
if stage in FREE:
    key, start, stop = FREE[stage]
    env["OPENROUTER_API_KEY"] = ENV[key]
    folder = "logs-mask-free-final" if start < 17 else "logs-mask-free-final-b"
    cmd = BASE + ["-T", "version=final", "-T", f"start={start}", "-T", f"stop={stop}", "--max-retries", "5", "--log-dir", folder]
else:
    env["ANTHROPIC_API_KEY"], env["ANTHROPIC_BASE_URL"] = ENV["ANTHROPIC_API_KEY"], "https://api.anthropic.com"
    if stage in ("smoke", "full"):
        version, folder = ("finalsmoke", "logs-mask-paid-finalsmoke") if stage == "smoke" else ("final", "logs-mask-paid-final")
        assert spent is not None and round(spent + FULL_N * CEILING_MAX, 2) <= BUDGET, \
            "smoke/full need --spent X (the paid spend so far), and X + the full run's hard ceiling ($72.15) must fit $96"
        assert stage == "smoke" or not (os.path.isdir(folder) and os.listdir(folder)), "the full run already has a log: use retry, never a second full run"
        cmd = PAID_BASE + ["-T", f"version={version}"] + PAID + ["--log-dir", folder]
    else:
        assert len(words) == 2 and spent is not None, "retry needs the log and --spent <paid spend so far>"
        log = read_eval_log(words[1])
        a = log.eval.task_args
        planned = len(a.get("arms") or range(5)) * ((a.get("stop") or 130) - (a.get("start") or 0))  # no arms arg = all 5 arms
        redo = (planned - len(log.samples)) + sum(s.error is not None for s in log.samples)
        cost = redo * CEILING_MAX
        print(f"errored or missing samples: {redo} -> retry ceiling ${cost:.2f}; money left ${BUDGET - spent:.2f}")
        assert cost <= BUDGET - spent, "the retry does not fit the money left: report the run as it is"
        cmd = ["eval-retry", words[1], "--max-connections", "8", "--log-dir", "logs-mask-paid-final-retry"]
print(INSPECT, " ".join(cmd))
if "--go" in argv:
    sys.exit(subprocess.run([INSPECT, *cmd], env=env).returncode)
