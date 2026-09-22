"""
Central policy version registry.

Every generated signal is stamped with the version of each policy axis that
produced it, so learning, analytics, and forward tests can attribute outcomes
to the configuration that was actually live - instead of pooling results from
materially different strategies under one prompt tag.

Bump the matching constant (use the date of the change) whenever behavior
changes on that axis:

- STRATEGY: signal selection, entry policy, target/stop calibration, exits,
  conviction tiering, validity windows.
- PROMPT: the analysis prompt template itself, or the model/provider used to
  answer it (optimization directives are tracked separately by the
  prompt-optimization service and appended to the composed version).
- EXECUTION: order placement, protective SL/TP structure, runner logic,
  position monitoring behavior.
- LEARNING: learning weights, offline policy probability, schedule selection,
  sizing adjustments derived from history.

Forward tests freeze the bundle version at start; a bundle change mid-test
means the test is no longer clean and should be restarted.
"""

from typing import Optional

STRATEGY_POLICY_VERSION = "strategy_2026-07-27-ev-deterministic-risk"
PROMPT_POLICY_VERSION = "prompt_2026-07-19"
EXECUTION_POLICY_VERSION = "execution_2026-08-03-revalidated-entry-v2"
LEARNING_POLICY_VERSION = "learning_2026-07-27-calibrated-net-ev"


def policy_stamp() -> dict:
    """Per-axis versions, stored on each signal for outcome attribution."""
    return {
        "strategy": STRATEGY_POLICY_VERSION,
        "prompt": PROMPT_POLICY_VERSION,
        "execution": EXECUTION_POLICY_VERSION,
        "learning": LEARNING_POLICY_VERSION,
    }


def policy_bundle_version() -> str:
    """Single comparable identifier for the full live configuration."""
    return "+".join(
        (
            STRATEGY_POLICY_VERSION,
            PROMPT_POLICY_VERSION,
            EXECUTION_POLICY_VERSION,
            LEARNING_POLICY_VERSION,
        )
    )


def compose_prompt_version(optimization_version: Optional[str]) -> str:
    """
    Combine the template-level prompt version with the prompt-optimization
    service's directive version. Analytics buckets split whenever either
    changes, which is the intent - a template rewrite is a different prompt
    even if the optimization directives kept their number.
    """
    base = PROMPT_POLICY_VERSION
    opt = str(optimization_version or "").strip()
    return f"{base}/{opt}" if opt else base
