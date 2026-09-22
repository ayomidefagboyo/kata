"""Shared trust rules for using modeled signal edge in live decisions."""

from typing import Any, Dict


def reliable_negative_edge_veto(
    edge_estimate: Any,
    *,
    threshold_pct: float = 0.0,
    minimum_samples: int = 20,
) -> bool:
    """Return whether a negative EV estimate is reliable enough to hard-veto.

    Cold-start EV is derived from heavily shrunk LLM confidence and estimated
    fill probability. It is useful for ranking, but it is not calibrated enough
    to erase an otherwise accepted directional signal. A hard veto is reserved
    for a validated production policy with a meaningful evidence sample.
    """
    if not isinstance(edge_estimate, dict):
        return False
    if edge_estimate.get("probability_source") != "calibrated_logistic_policy":
        return False
    if str(edge_estimate.get("policy_validation_status") or "").lower() != "passed":
        return False
    try:
        evidence_samples = int(edge_estimate.get("evidence_samples") or 0)
        net_ev_pct = float(edge_estimate.get("net_expected_value_pct"))
    except (TypeError, ValueError):
        return False
    return evidence_samples >= max(1, int(minimum_samples)) and net_ev_pct <= float(
        threshold_pct
    )
