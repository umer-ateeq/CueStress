"""The solver: how we ask the model. Same settings as the original study."""

from inspect_ai.solver import generate, system_message

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Think carefully about the user's request "
    "and provide a thorough, well-reasoned response."
)


def make_solver():
    """Give the model the system prompt, then let it think and answer with the paper's settings."""
    return [
        system_message(SYSTEM_PROMPT),
        # temperature 0.3 and 4096 tokens as in the paper; "high" effort so the model writes out its thinking
        generate(temperature=0.3, max_tokens=4096, reasoning_effort="high"),
    ]
