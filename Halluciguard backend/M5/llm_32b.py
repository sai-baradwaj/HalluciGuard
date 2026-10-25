"""
32B repair/verification model wrapper.

CONFIRMED BUG (fixed here): generate_32b() previously called the local
"qwen3:8b" Ollama tag instead of the intended 32B model, silently making
the Repair Agent and Fact Verifier run on the 8B model while claiming to
use a 32B model. This file restores the intended, configurable 32B model
and fails loudly (instead of silently falling back to 8B) when that model
is not available.
"""

import os

import ollama


# The 32B model tag is configurable via the project's existing .env /
# environment-variable convention (see config.py / python-dotenv usage
# elsewhere in this backend). Default matches the Ollama tag naming used
# for Qwen3 32B ("qwen3:32b"). Update HALLUCIGUARD_REPAIR_MODEL in .env
# if your local Ollama install uses a different tag.
REPAIR_MODEL = os.getenv("HALLUCIGUARD_REPAIR_MODEL", "qwen3:32b")


class Model32BUnavailableError(RuntimeError):
    """Raised when the configured 32B Ollama model cannot be used.

    This is intentionally NOT caught-and-downgraded to the 8B model:
    doing so would silently weaken fact verification and repair quality
    while reporting success, which is explicitly disallowed.
    """


def generate_32b(
    prompt,
    max_new_tokens=500,
    temperature=0.5,
):
    """
    Generate a response using the configured 32B Ollama model.

    Used for:
    - Fact verification
    - Repair agent

    Raises:
        Model32BUnavailableError: if the configured 32B model is missing
        or the Ollama call otherwise fails. Callers must not treat this
        as a successful verification/repair.
    """

    try:
        response = ollama.chat(
            model=REPAIR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt.replace("\n/no_think", ""),
                }
            ],
            think=False,
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        )
    except Exception as exc:
        message = str(exc).lower()

        if "not found" in message or "no such model" in message:
            raise Model32BUnavailableError(
                f"The configured 32B repair model '{REPAIR_MODEL}' is not "
                f"installed in Ollama. Run `ollama pull {REPAIR_MODEL}` or "
                f"set HALLUCIGUARD_REPAIR_MODEL to an installed model tag. "
                f"Refusing to silently fall back to the 8B model."
            ) from exc

        raise Model32BUnavailableError(
            f"The 32B repair model '{REPAIR_MODEL}' failed to respond: {exc}"
        ) from exc

    content = (response.message.content or "").strip()

    if not content:
        raise Model32BUnavailableError(
            f"The 32B repair model '{REPAIR_MODEL}' returned an empty response."
        )

    return content
