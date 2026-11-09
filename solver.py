"""The solver: the target model answers each version as it is, with no system prompt of ours (MASK's own system prompt is part
of its samples)."""
from inspect_ai.solver import generate


def make_solver(effort=None, max_tokens=4096):
    """One reply per sample. effort: the target's reasoning effort (Claude: medium); max_tokens: its reply limit."""
    return generate(reasoning_effort=effort, max_tokens=max_tokens)
