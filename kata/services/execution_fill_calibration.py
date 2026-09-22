"""Production fill-probability calibration from Yuki Hyperliquid outcomes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kata.services.temporal_model_validation import (
    build_purged_walk_forward_folds,
    outcome_known_at,
    parse_utc_datetime,
    probability_validation_metrics,
    validation_passed,
)

logger = logging.getLogger(__name__)

FEATURE_VERSION = "hyperliquid_fill_v1"


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _setup_bucket(setup_type: Any, entry_strategy: Any) -> str:
    combined = f"{setup_type or ''} {entry_strategy or ''}".lower()
    if "market" in combined:
        return "market"
    if any(marker in combined for marker in ("breakout", "continuation", "momentum")):
        return "continuation"
    if any(marker in combined for marker in ("pullback", "retracement", "limit")):
        return "pullback"
    if any(marker in combined for marker in ("mean", "reversion", "fade")):
        return "mean_reversion"
    return "other"


def fill_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """Encode all execution-time features with stable, bounded transforms."""
    distance = max(0.0, _number(features.get("entry_distance_pct"), 0.0))
    age = max(0.0, _number(features.get("signal_age_hours"), 0.0))
    spread = max(0.0, _number(features.get("spread_pct"), 0.08))
    depth = max(1.0, _number(features.get("depth_usd"), 25_000.0))
    volatility = max(0.0, _number(features.get("volatility_pct"), 2.0))
    side = str(features.get("side") or "LONG").upper()
    setup = _setup_bucket(features.get("setup_type"), features.get("entry_strategy"))
    setup_order = ("market", "continuation", "pullback", "mean_reversion", "other")
    return np.asarray([
        min(distance, 25.0),
        min(age, 168.0),
        min(spread, 5.0),
        math.log10(min(depth, 1_000_000_000.0)),
        min(volatility, 50.0),
        1.0 if side in {"SHORT", "SELL"} else 0.0,
        *(1.0 if setup == name else 0.0 for name in setup_order),
    ], dtype=float)


def features_from_trade(row: Dict[str, Any]) -> Optional[Tuple[np.ndarray, int]]:
    """Extract one terminal Hyperliquid order and its actual fill outcome."""
    status = str(row.get("status") or "").lower()
    filled = bool(row.get("filled_at")) or status in {"filled", "active", "closed"}
    if not filled and status not in {"cancelled", "expired"}:
        return None

    metadata = row.get("trade_metadata") or row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    edge = metadata.get("edge_estimate") or {}
    structure = metadata.get("market_structure_context") or {}
    order_flow = structure.get("order_flow") or {}
    orderbook = order_flow.get("orderbook") or {}
    bid_depth = _number(
        orderbook.get("bid_depth_usdt") or orderbook.get("bid_depth_usd"),
        0.0,
    )
    ask_depth = _number(
        orderbook.get("ask_depth_usdt") or orderbook.get("ask_depth_usd"),
        0.0,
    )
    positive_depths = [depth for depth in (bid_depth, ask_depth) if depth > 0]
    depth = min(positive_depths) if positive_depths else 25_000.0
    features = {
        "entry_distance_pct": metadata.get("entry_distance_pct_at_order"),
        "signal_age_hours": metadata.get("entry_age_hours_at_order"),
        "spread_pct": orderbook.get("spread_pct") or orderbook.get("spread_percentage"),
        "depth_usd": depth,
        "volatility_pct": edge.get("atr_pct") or metadata.get("atr_pct_at_order"),
        "side": row.get("side") or metadata.get("signal_direction"),
        "setup_type": edge.get("setup_type") or metadata.get("setup_type"),
        "entry_strategy": edge.get("entry_strategy") or metadata.get("entry_strategy"),
    }
    return fill_feature_vector(features), int(filled)


@dataclass
class ExecutionFillModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    sample_count: int
    fill_count: int
    trained_at: datetime
    validation_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_fill_rate(self) -> float:
        return self.fill_count / max(self.sample_count, 1)

    def predict(self, features: Dict[str, Any]) -> float:
        vector = fill_feature_vector(features).reshape(1, -1)
        raw = float(self.classifier.predict_proba(self.scaler.transform(vector))[0, 1])
        # Small production datasets are useful but should not create false certainty.
        reliability = self.sample_count / (self.sample_count + 40.0)
        calibrated = reliability * raw + (1.0 - reliability) * self.base_fill_rate
        return max(0.02, min(0.98, calibrated))


class ExecutionFillCalibrator:
    """Train and serve a lightweight model from completed Yuki orders."""

    def __init__(self, minimum_samples: int = 30):
        self.minimum_samples = max(20, int(minimum_samples or 30))
        self.model: Optional[ExecutionFillModel] = None

    def fit(self, rows: Sequence[Dict[str, Any]]) -> Optional[ExecutionFillModel]:
        examples = []
        fallback_origin = datetime(2000, 1, 1, tzinfo=timezone.utc)
        for index, row in enumerate(rows):
            item = features_from_trade(row)
            if item is None:
                continue
            raw_observed = parse_utc_datetime(row.get("created_at"))
            observed = raw_observed or fallback_origin + timedelta(days=index * 2)
            metadata = row.get("trade_metadata") or row.get("metadata") or {}
            explicit_outcome = (
                row.get("filled_at")
                or row.get("closed_at")
                or (metadata.get("entry_order_expires_at") if isinstance(metadata, dict) else None)
            )
            known = outcome_known_at(
                observed,
                explicit_outcome if raw_observed is not None else None,
                fallback_purge_hours=12.0,
            )
            examples.append((observed, known, item[0], item[1]))
        if len(examples) < self.minimum_samples:
            logger.info(
                "Fill calibration waiting for samples: %d/%d terminal Yuki orders",
                len(examples),
                self.minimum_samples,
            )
            return None
        examples.sort(key=lambda example: example[0])
        observed_at = [example[0] for example in examples]
        known_at = [example[1] for example in examples]
        matrix = np.vstack([example[2] for example in examples])
        labels = np.asarray([example[3] for example in examples], dtype=int)
        if len(set(labels.tolist())) < 2:
            logger.warning("Fill calibration requires both filled and unfilled orders")
            return None

        validation_metrics = self._walk_forward_validation(
            matrix,
            labels,
            observed_at,
            known_at,
        )
        if validation_metrics.get("status") != "passed":
            logger.warning(
                "Fill model candidate not promoted: temporal validation status=%s samples=%s brier=%s baseline=%s",
                validation_metrics.get("status"),
                validation_metrics.get("validation_samples", 0),
                validation_metrics.get("brier_score"),
                validation_metrics.get("baseline_brier_score"),
            )
            return None

        scaler, classifier = self._fit_estimator(matrix, labels)
        self.model = ExecutionFillModel(
            scaler=scaler,
            classifier=classifier,
            sample_count=len(labels),
            fill_count=int(labels.sum()),
            trained_at=datetime.now(timezone.utc),
            validation_metrics=validation_metrics,
        )
        logger.info(
            "Trained %s on %d Hyperliquid outcomes (actual fill rate %.1f%%)",
            FEATURE_VERSION,
            self.model.sample_count,
            self.model.base_fill_rate * 100.0,
        )
        return self.model

    @staticmethod
    def _fit_estimator(
        matrix: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[StandardScaler, LogisticRegression]:
        scaler = StandardScaler().fit(matrix)
        classifier = LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ).fit(scaler.transform(matrix), labels)
        return scaler, classifier

    def _walk_forward_validation(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        observed_at: Sequence[datetime],
        known_at: Sequence[datetime],
    ) -> Dict[str, Any]:
        folds = build_purged_walk_forward_folds(
            observed_at,
            known_at,
            min_train_size=12,
            min_validation_size=4,
            max_splits=3,
        )
        predictions: List[float] = []
        baselines: List[float] = []
        outcomes: List[int] = []
        purged_count = 0
        cutoffs: List[str] = []

        for fold in folds:
            train_labels = labels[fold.train_indices]
            if len(np.unique(train_labels)) < 2:
                continue
            scaler, classifier = self._fit_estimator(
                matrix[fold.train_indices],
                train_labels,
            )
            fold_probabilities = classifier.predict_proba(
                scaler.transform(matrix[fold.validation_indices])
            )[:, 1]
            validation_labels = labels[fold.validation_indices]
            predictions.extend(fold_probabilities.tolist())
            outcomes.extend(validation_labels.tolist())
            baselines.extend([float(np.mean(train_labels))] * len(validation_labels))
            purged_count += fold.purged_count
            cutoffs.append(fold.validation_start.isoformat())

        metrics = probability_validation_metrics(outcomes, predictions, baselines)
        if not metrics:
            return {
                "status": "insufficient_chronological_history",
                "folds": 0,
                "validation_samples": 0,
            }
        metrics.update({
            "status": "passed" if validation_passed(metrics) else "failed",
            "folds": len(cutoffs),
            "validation_samples": len(outcomes),
            "purged_samples": purged_count,
            "validation_cutoffs": cutoffs,
            "method": "purged_expanding_walk_forward_v1",
        })
        return metrics

    def train_from_supabase(self, db: Any, lookback_days: int = 90) -> Optional[ExecutionFillModel]:
        since = (datetime.now(timezone.utc) - timedelta(days=max(7, lookback_days))).isoformat()
        rows = (
            db.table("agent_trades")
            .select("status,side,filled_at,closed_at,created_at,trade_metadata")
            .eq("agent_type", "yuki")
            .in_("status", ["filled", "active", "closed", "cancelled", "expired"])
            .gte("created_at", since)
            .order("created_at", desc=True)
            .limit(2000)
            .execute()
        ).data or []
        return self.fit(rows)

    def predict(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if self.model is None:
            # Conservative cold-start prior, not a distance/ATR theory or hard floor.
            setup = _setup_bucket(features.get("setup_type"), features.get("entry_strategy"))
            probability = 0.98 if setup == "market" else 0.35
            return probability, {
                "fill_probability_source": "conservative_cold_start",
                "fill_model_version": FEATURE_VERSION,
                "fill_evidence_samples": 0,
                "fill_validation_status": "unavailable",
            }
        validation = self.model.validation_metrics or {}
        return self.model.predict(features), {
            "fill_probability_source": "trained_hyperliquid_logistic",
            "fill_model_version": FEATURE_VERSION,
            "fill_evidence_samples": self.model.sample_count,
            "observed_fill_rate": round(self.model.base_fill_rate, 6),
            "fill_model_trained_at": self.model.trained_at.isoformat(),
            "fill_validation_status": validation.get("status"),
            "fill_validation_samples": validation.get("validation_samples", 0),
            "fill_validation_brier_score": validation.get("brier_score"),
            "fill_validation_baseline_brier_score": validation.get("baseline_brier_score"),
        }
