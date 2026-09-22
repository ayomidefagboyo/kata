"""Shared chronological validation helpers for production-learned models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PurgedWalkForwardFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_start: datetime
    purged_count: int


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def outcome_known_at(
    observed_at: datetime,
    explicit_outcome_at: Any,
    *,
    fallback_purge_hours: float,
) -> datetime:
    """Return the earliest safe time at which an example's outcome was knowable."""
    explicit = parse_utc_datetime(explicit_outcome_at)
    if explicit is not None and explicit >= observed_at:
        return explicit
    return observed_at + timedelta(hours=max(0.0, float(fallback_purge_hours)))


def build_purged_walk_forward_folds(
    observed_at: Sequence[datetime],
    outcome_at: Sequence[datetime],
    *,
    min_train_size: int,
    min_validation_size: int,
    max_splits: int = 3,
) -> List[PurgedWalkForwardFold]:
    """Build expanding time folds and remove training labels not known at each cutoff."""
    sample_count = len(observed_at)
    if sample_count != len(outcome_at) or sample_count < min_train_size + min_validation_size:
        return []

    validation_size = max(
        int(min_validation_size),
        int(math.ceil(sample_count * 0.15)),
    )
    possible_splits = max(1, (sample_count - min_train_size) // validation_size)
    split_count = min(max(1, int(max_splits)), possible_splits)
    first_validation = sample_count - split_count * validation_size
    first_validation = max(int(min_train_size), first_validation)

    folds: List[PurgedWalkForwardFold] = []
    validation_start_index = first_validation
    while validation_start_index < sample_count and len(folds) < split_count:
        validation_end_index = min(sample_count, validation_start_index + validation_size)
        if validation_end_index - validation_start_index < min_validation_size:
            break

        validation_start = observed_at[validation_start_index]
        eligible_train = [
            index
            for index in range(validation_start_index)
            if outcome_at[index] < validation_start
        ]
        if len(eligible_train) >= min_train_size:
            folds.append(PurgedWalkForwardFold(
                train_indices=np.asarray(eligible_train, dtype=int),
                validation_indices=np.arange(
                    validation_start_index,
                    validation_end_index,
                    dtype=int,
                ),
                validation_start=validation_start,
                purged_count=validation_start_index - len(eligible_train),
            ))
        validation_start_index = validation_end_index
    return folds


def probability_validation_metrics(
    outcomes: Sequence[int],
    probabilities: Sequence[float],
    baseline_probabilities: Sequence[float],
) -> Dict[str, float]:
    labels = np.asarray(outcomes, dtype=float)
    predicted = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    baseline = np.clip(
        np.asarray(baseline_probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    if labels.size == 0 or labels.size != predicted.size or labels.size != baseline.size:
        return {}

    brier = float(np.mean(np.square(predicted - labels)))
    baseline_brier = float(np.mean(np.square(baseline - labels)))
    log_loss = float(-np.mean(labels * np.log(predicted) + (1.0 - labels) * np.log(1.0 - predicted)))
    baseline_log_loss = float(
        -np.mean(labels * np.log(baseline) + (1.0 - labels) * np.log(1.0 - baseline))
    )
    return {
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_score": (
            1.0 - brier / baseline_brier if baseline_brier > 0 else 0.0
        ),
        "log_loss": log_loss,
        "baseline_log_loss": baseline_log_loss,
        "mean_prediction": float(np.mean(predicted)),
        "observed_rate": float(np.mean(labels)),
    }


def validation_passed(metrics: Dict[str, float], *, tolerance: float = 0.005) -> bool:
    """Require the candidate to be no worse than a historical-rate baseline."""
    if not metrics:
        return False
    return bool(
        metrics["brier_score"] <= metrics["baseline_brier_score"] + tolerance
        and metrics["log_loss"] <= metrics["baseline_log_loss"] + tolerance
    )
