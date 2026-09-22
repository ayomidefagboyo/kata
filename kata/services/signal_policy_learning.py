"""
Offline-trained signal policy: P(win | features, direction) from resolved platform signals.

Used as the predictive layer; LLM remains synthesis + guardrails in the ensemble.
"""
from __future__ import annotations

import json
import logging
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from kata.services.temporal_model_validation import (
    build_purged_walk_forward_folds,
    outcome_known_at,
    parse_utc_datetime,
    probability_validation_metrics,
    validation_passed,
)

logger = logging.getLogger(__name__)

POLICY_FEATURE_VERSION = "v1"
# Length of flattened vector from build_policy_feature_vector (keep in sync when adding features).
POLICY_FEATURE_DIM = 25
POLICY_TRAINING_JSON_KEY = "policy_training"

REGIME_BUCKETS: Tuple[str, ...] = (
    "strong_uptrend",
    "strong_downtrend",
    "uptrend",
    "downtrend",
    "ranging",
    "unknown",
)


def default_policy_bundle_path() -> Path:
    backend_root = Path(__file__).resolve().parent.parent.parent
    return backend_root / "data" / "signal_policy" / "policy_bundle.joblib"


def normalize_regime_bucket(regime_type: str) -> str:
    r = (regime_type or "").lower().strip()
    if r in REGIME_BUCKETS:
        return r
    if "strong" in r and ("up" in r or "uptrend" in r):
        return "strong_uptrend"
    if "strong" in r and ("down" in r or "downtrend" in r):
        return "strong_downtrend"
    if "range" in r or "sideways" in r:
        return "ranging"
    if "up" in r or "bull" in r:
        return "uptrend"
    if "down" in r or "bear" in r:
        return "downtrend"
    return "unknown"


def _norm_opp_score(v: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.5
    if x > 1.0:
        x = x / 100.0
    return max(0.0, min(1.0, x))


def build_policy_feature_vector(
    *,
    rsi_14: float,
    bb_position: float,
    macd_histogram: float,
    momentum_score: float,
    strength_score: float,
    stoch_k: float,
    trend_alignment: float,
    price_efficiency: float,
    volatility_24h: float,
    atr_pct: float,
    opp_overall: float,
    opp_volume: float,
    opp_momentum: float,
    opp_trend: float,
    opp_institutional: float,
    rule_confidence: float,
    eval_direction: str,
    regime_type: str,
    signal_confidence: float,
) -> np.ndarray:
    """Fixed-order feature vector; must stay aligned with training."""
    d = (eval_direction or "HOLD").upper()
    is_long = 1.0 if d == "LONG" else 0.0
    is_short = 1.0 if d == "SHORT" else 0.0
    bucket = normalize_regime_bucket(regime_type)
    reg = [1.0 if bucket == b else 0.0 for b in REGIME_BUCKETS]

    rc = max(0.0, min(1.0, float(rule_confidence)))
    sc = max(0.0, min(1.0, float(signal_confidence)))

    vec = np.array(
        [
            float(rsi_14) / 100.0,
            max(0.0, min(1.0, float(bb_position))),
            float(np.tanh(float(macd_histogram))),
            max(0.0, min(1.0, float(momentum_score))),
            max(0.0, min(1.0, float(strength_score))),
            float(stoch_k) / 100.0,
            max(0.0, min(1.0, float(trend_alignment))),
            max(0.0, min(1.0, float(price_efficiency))),
            max(0.0, min(1.0, float(volatility_24h) / 100.0)),
            max(0.0, min(1.0, float(atr_pct))),
            _norm_opp_score(opp_overall),
            max(0.0, min(1.0, float(opp_volume))),
            max(0.0, min(1.0, float(opp_momentum))),
            max(0.0, min(1.0, float(opp_trend))),
            max(0.0, min(1.0, float(opp_institutional))),
            rc,
            is_long,
            is_short,
            sc,
            *reg,
        ],
        dtype=np.float64,
    )
    return vec.reshape(1, -1)


def policy_training_snapshot_from_live(
    ta: Any,
    opp: Any,
    rule_confidence: float,
    regime_type: str,
    eval_direction: str,
    signal_confidence: float,
) -> Dict[str, Any]:
    """
    Serialize the exact feature vector used at inference time for later training.

    `ta` / `opp` are TechnicalAnalysis / OpportunityScore (duck-typed to avoid import cycles).
    """
    price = max(float(getattr(ta, "current_price", 0.0) or 0.0), 1e-12)
    atr_pct = min(1.0, float(getattr(ta, "atr_14", 0.0) or 0.0) / price * 100.0)
    arr = build_policy_feature_vector(
        rsi_14=float(getattr(ta, "rsi_14", 50.0)),
        bb_position=float(getattr(ta, "bb_position", 0.5)),
        macd_histogram=float(getattr(ta, "macd_histogram", 0.0)),
        momentum_score=float(getattr(ta, "momentum_score", 0.5)),
        strength_score=float(getattr(ta, "strength_score", 0.5)),
        stoch_k=float(getattr(ta, "stoch_k", 50.0)),
        trend_alignment=float(getattr(ta, "trend_alignment", 0.5)),
        price_efficiency=float(getattr(ta, "price_efficiency", 0.5)),
        volatility_24h=float(getattr(ta, "volatility_24h", 0.0)),
        atr_pct=atr_pct,
        opp_overall=float(getattr(opp, "overall_score", 0.5)),
        opp_volume=float(getattr(opp, "volume_score", 0.5)),
        opp_momentum=float(getattr(opp, "momentum_score", 0.5)),
        opp_trend=float(getattr(opp, "trend_score", 0.5)),
        opp_institutional=float(getattr(opp, "institutional_score", 0.5)),
        rule_confidence=float(rule_confidence),
        eval_direction=eval_direction,
        regime_type=regime_type,
        signal_confidence=float(signal_confidence),
    )
    flat = arr.reshape(-1)
    if flat.shape[0] != POLICY_FEATURE_DIM:
        logger.warning(
            "Policy feature length %s != POLICY_FEATURE_DIM %s — update constant",
            flat.shape[0],
            POLICY_FEATURE_DIM,
        )
    return {
        "version": POLICY_FEATURE_VERSION,
        "dim": int(flat.shape[0]),
        "values": flat.tolist(),
        "eval_direction": str(eval_direction or "").upper(),
    }


def feature_array_from_policy_training_blob(blob: Any) -> Optional[np.ndarray]:
    """If JSON snapshot is present and valid, return (1, n_features); else None."""
    if not isinstance(blob, dict):
        return None
    ver = str(blob.get("version") or "")
    values = blob.get("values")
    if ver != POLICY_FEATURE_VERSION or not isinstance(values, list):
        return None
    try:
        arr = np.array([float(x) for x in values], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != POLICY_FEATURE_DIM:
        return None
    return arr.reshape(1, -1)


def _float_from(d: Dict[str, Any], key: str, default: float) -> float:
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def build_policy_feature_vector_from_signal_row(row: Dict[str, Any]) -> Optional[np.ndarray]:
    """Reconstruct features from a platform_signals row (JSON fields)."""
    direction = str(row.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None

    br = row.get("ai_confidence_breakdown") or {}
    if isinstance(br, str):
        try:
            br = json.loads(br)
        except json.JSONDecodeError:
            br = {}
    if not isinstance(br, dict):
        br = {}

    snap = br.get(POLICY_TRAINING_JSON_KEY)
    from_snapshot = feature_array_from_policy_training_blob(snap)
    if from_snapshot is not None:
        return from_snapshot

    ti = row.get("technical_indicators") or {}
    if not isinstance(ti, dict):
        ti = {}
    mc = row.get("market_conditions") or {}
    if not isinstance(mc, dict):
        mc = {}

    regime_type = str(br.get("market_regime") or mc.get("trend_direction") or "")

    rsi = _float_from(ti, "rsi_14", 50.0)
    bb = _float_from(ti, "bb_position", 0.5)
    macd_h = _float_from(ti, "macd_histogram", 0.0)
    mom = _float_from(ti, "momentum_score", 0.5)
    strength = _float_from(ti, "strength_score", 0.5)

    price = _float_from(mc, "current_price", 0.0) or 1.0
    vol24 = _float_from(mc, "volatility_24h", 1.0)
    pc24 = _float_from(mc, "price_change_24h", 0.0)

    stoch_k = 50.0
    trend_align = 0.5
    price_eff = 0.5
    atr_pct = 0.01

    conf = _float_from(row, "confidence", 0.6)
    overall = _float_from(row, "overall_score", 0.5)

    return build_policy_feature_vector(
        rsi_14=rsi,
        bb_position=bb,
        macd_histogram=macd_h,
        momentum_score=mom,
        strength_score=strength,
        stoch_k=stoch_k,
        trend_alignment=trend_align,
        price_efficiency=price_eff,
        volatility_24h=vol24,
        atr_pct=min(1.0, abs(pc24) / 100.0 + 0.001),
        opp_overall=overall,
        opp_volume=0.5,
        opp_momentum=mom,
        opp_trend=0.5 if "bull" in str(mc.get("trend_direction", "")).lower() else 0.5,
        opp_institutional=0.5,
        rule_confidence=conf,
        eval_direction=direction,
        regime_type=regime_type,
        signal_confidence=conf,
    )


# Pseudo-observations at a neutral 50% prior used to shrink the classifier's
# raw probability toward 0.5 when the training sample is small. A logistic
# regression fit on tens of resolved signals will happily emit 0.95s that the
# sample size cannot support; with n=60 real samples a raw 0.95 shrinks to
# ~0.75, and the shrinkage fades as the training set actually grows.
POLICY_PROBABILITY_PRIOR_PSEUDOCOUNT = 50.0


@dataclass
class SignalPolicyBundle:
    pipeline: Any
    feature_version: str
    trained_at: str
    n_samples: int
    win_rate: float
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    def predict_win_probability(self, features: np.ndarray) -> float:
        proba = self.pipeline.predict_proba(features)[0]
        classes = list(getattr(self.pipeline, "classes_", []))
        if not classes:
            clf = self.pipeline.named_steps.get("clf") if hasattr(self.pipeline, "named_steps") else None
            classes = list(getattr(clf, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
            raw = float(proba[idx])
        elif len(proba) >= 2:
            raw = float(proba[1])
        else:
            raw = float(proba[0])
        return self._shrink_toward_prior(raw)

    def _shrink_toward_prior(self, raw: float) -> float:
        """Sample-size-aware calibration: shrink toward a neutral 0.5 prior."""
        raw = max(0.0, min(1.0, raw))
        n = max(0, int(self.n_samples or 0))
        k = POLICY_PROBABILITY_PRIOR_PSEUDOCOUNT
        shrunk = (n * raw + k * 0.5) / (n + k) if (n + k) > 0 else 0.5
        return float(max(0.0, min(1.0, shrunk)))


def train_policy_bundle(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if X.size == 0 or len(y) < 10:
        raise ValueError("Insufficient samples for policy training")

    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    calibration_splits = _chronological_calibration_splits(np.asarray(y, dtype=int))
    if calibration_splits:
        # Sigmoid calibration is deliberately preferred over isotonic here:
        # Yuki's resolved sample is still small and isotonic would overfit. The
        # expanding folds also prevent later regimes from calibrating earlier ones.
        calibrated = CalibratedClassifierCV(
            estimator=base_pipeline,
            method="sigmoid",
            cv=calibration_splits,
            n_jobs=1,
        )
        # CalibratedClassifierCV cannot reliably route Pipeline step-specific
        # sample weights across sklearn versions. Class balancing remains in
        # the estimator; cross-validated calibration takes precedence.
        calibrated.fit(X, y)
        return calibrated

    # Degenerate small-class fallback. The bundle still applies empirical-Bayes
    # shrinkage at inference and cannot become an EV bypass until n >= 20.
    if sample_weight is not None and len(sample_weight) == len(y):
        base_pipeline.fit(X, y, clf__sample_weight=sample_weight)
    else:
        base_pipeline.fit(X, y)
    return base_pipeline


def _chronological_calibration_splits(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create expanding, class-complete folds in the order examples occurred."""
    sample_count = len(y)
    if sample_count < 16:
        return []

    validation_size = max(4, sample_count // 5)
    first_validation = max(10, sample_count - validation_size * 3)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for validation_start in range(first_validation, sample_count, validation_size):
        validation_end = min(sample_count, validation_start + validation_size)
        # Leave one observation between train and calibration windows. The
        # outer walk-forward evaluation performs the timestamp-based purge.
        train_indices = np.arange(0, max(0, validation_start - 1), dtype=int)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        if len(train_indices) < 10 or len(validation_indices) < 4:
            continue
        if len(np.unique(y[train_indices])) < 2 or len(np.unique(y[validation_indices])) < 2:
            continue
        splits.append((train_indices, validation_indices))
    return splits


def _positive_class_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else min(1, probabilities.shape[1] - 1)
    return probabilities[:, positive_index]


def _walk_forward_policy_validation(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
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
    used_folds = 0
    purged_count = 0
    cutoffs: List[str] = []

    for fold in folds:
        train_y = y[fold.train_indices]
        validation_y = y[fold.validation_indices]
        if len(np.unique(train_y)) < 2:
            continue
        candidate = train_policy_bundle(
            X[fold.train_indices],
            train_y,
            sample_weight=weights[fold.train_indices],
        )
        fold_predictions = _positive_class_probabilities(
            candidate,
            X[fold.validation_indices],
        )
        predictions.extend(fold_predictions.tolist())
        outcomes.extend(validation_y.tolist())
        baselines.extend([float(np.mean(train_y))] * len(validation_y))
        purged_count += fold.purged_count
        cutoffs.append(fold.validation_start.isoformat())
        used_folds += 1

    metrics = probability_validation_metrics(outcomes, predictions, baselines)
    if not metrics:
        return {
            "status": "insufficient_chronological_history",
            "folds": 0,
            "validation_samples": 0,
        }
    metrics.update({
        "status": "passed" if validation_passed(metrics) else "failed",
        "folds": used_folds,
        "validation_samples": len(outcomes),
        "purged_samples": purged_count,
        "validation_cutoffs": cutoffs,
        "method": "purged_expanding_walk_forward_v1",
    })
    return metrics


def save_policy_bundle(
    path: Path,
    pipeline: Any,
    n_samples: int,
    win_rate: float,
    validation_metrics: Optional[Dict[str, Any]] = None,
    *,
    db: Any = None,
    agent_type: Optional[str] = None,
) -> None:
    import io
    trained_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "pipeline": pipeline,
        "feature_version": POLICY_FEATURE_VERSION,
        "trained_at": trained_at,
        "n_samples": int(n_samples),
        "win_rate": float(win_rate),
        "validation_metrics": dict(validation_metrics or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    logger.info("Saved signal policy bundle to %s (n=%s, win_rate=%.3f)", path, n_samples, win_rate)

    if db is not None and agent_type:
        try:
            buf = io.BytesIO()
            joblib.dump(payload, buf)
            model_bytes = buf.getvalue()
            db.table("signal_policy_bundles").upsert(
                {
                    "agent_type": agent_type,
                    "feature_version": POLICY_FEATURE_VERSION,
                    "trained_at": trained_at,
                    "n_samples": int(n_samples),
                    "win_rate": float(win_rate),
                    "model_bytes": "\\x" + model_bytes.hex(),
                    "updated_at": trained_at,
                },
                on_conflict="agent_type",
            ).execute()
            logger.info("Policy bundle persisted to DB for agent=%s", agent_type)
        except Exception as exc:
            logger.warning("Failed to persist policy bundle to DB: %s", exc)


def load_policy_bundle(path: Path) -> Optional[SignalPolicyBundle]:
    if not path.is_file():
        return None
    try:
        raw = joblib.load(path)
        if isinstance(raw, dict) and "pipeline" in raw:
            fv = str(raw.get("feature_version") or "")
            if fv and fv != POLICY_FEATURE_VERSION:
                logger.warning("Policy bundle feature version %s != %s — still loading", fv, POLICY_FEATURE_VERSION)
            return SignalPolicyBundle(
                pipeline=raw["pipeline"],
                feature_version=fv or POLICY_FEATURE_VERSION,
                trained_at=str(raw.get("trained_at") or ""),
                n_samples=int(raw.get("n_samples") or 0),
                win_rate=float(raw.get("win_rate") or 0.0),
                validation_metrics=dict(raw.get("validation_metrics") or {}),
                path=path,
            )
    except Exception as e:
        logger.warning("Failed to load policy bundle from %s: %s", path, e)
    return None


def load_policy_bundle_from_db(db: Any, agent_type: str, path: Path) -> Optional[SignalPolicyBundle]:
    """Load policy bundle from signal_policy_bundles table; restore to file as side-effect."""
    import io

    def decode_model_bytes(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, list):
            return bytes(value)
        if isinstance(value, str):
            if value.startswith("\\x"):
                return bytes.fromhex(value[2:])
            try:
                return base64.b64decode(value, validate=True)
            except Exception:
                return value.encode("latin1")
        raise TypeError(f"Unsupported model_bytes payload type: {type(value).__name__}")

    try:
        result = (
            db.table("signal_policy_bundles")
            .select("model_bytes, feature_version, trained_at, n_samples, win_rate")
            .eq("agent_type", agent_type)
            .single()
            .execute()
        )
        if not result.data or not result.data.get("model_bytes"):
            return None
        raw_bytes = decode_model_bytes(result.data["model_bytes"])
        raw = joblib.load(io.BytesIO(raw_bytes))
        if not (isinstance(raw, dict) and "pipeline" in raw):
            return None
        # Restore file so future loads don't hit the DB
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(raw, path)
        logger.info("Policy bundle restored from DB to %s (agent=%s)", path, agent_type)
        fv = str(raw.get("feature_version") or POLICY_FEATURE_VERSION)
        return SignalPolicyBundle(
            pipeline=raw["pipeline"],
            feature_version=fv,
            trained_at=str(raw.get("trained_at") or ""),
            n_samples=int(raw.get("n_samples") or 0),
            win_rate=float(raw.get("win_rate") or 0.0),
            validation_metrics=dict(raw.get("validation_metrics") or {}),
            path=path,
        )
    except Exception as exc:
        logger.warning("Failed to load policy bundle from DB (agent=%s): %s", agent_type, exc)
    return None


def fetch_training_rows_from_supabase(
    supabase: Any,
    lookback_days: int,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    start = (datetime.now(timezone.utc) - timedelta(days=max(lookback_days, 14))).isoformat()
    common_fields = (
        "direction,status,confidence,overall_score,technical_indicators,"
        "market_conditions,ai_confidence_breakdown,entry_price,leverage,created_at,expires_at"
    )
    terminal_statuses = ["hit_target_1", "hit_target_2", "hit_stop_loss"]

    # Primary path: current platform_signals schema
    try:
        res = (
            supabase.table("platform_signals")
            .select(f"{common_fields},stop_loss,target_1,target_2")
            .gte("created_at", start)
            .in_("status", terminal_statuses)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(res.data or [])
    except Exception as modern_schema_error:
        logger.warning(
            "Policy training query failed on modern columns, retrying legacy aliases: %s",
            modern_schema_error,
        )

    # Legacy fallback path: older deployments that still use *_price columns
    res = (
        supabase.table("platform_signals")
        .select(f"{common_fields},stop_loss_price,target_1_price,target_2_price")
        .gte("created_at", start)
        .in_("status", terminal_statuses)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = list(res.data or [])
    for row in rows:
        row["stop_loss"] = row.get("stop_loss_price")
        row["target_1"] = row.get("target_1_price")
        row["target_2"] = row.get("target_2_price")
    return rows


def _compute_leveraged_pnl_pct(row: Dict[str, Any]) -> float:
    """Estimate leveraged PnL % from signal row outcome (sign carries direction of P&L).

    Returns positive value for wins (capped at +60%), negative for losses (capped at -30%).
    Falls back to ±10% if price fields are missing.
    """
    status = str(row.get("status") or "").lower()
    direction = str(row.get("direction") or "").upper()
    try:
        entry = float(row.get("entry_price") or 0.0)
        sl = float((row.get("stop_loss") if row.get("stop_loss") is not None else row.get("stop_loss_price")) or 0.0)
        t1 = float((row.get("target_1") if row.get("target_1") is not None else row.get("target_1_price")) or 0.0)
        t2 = float((row.get("target_2") if row.get("target_2") is not None else row.get("target_2_price")) or 0.0)
        lev = max(1.0, float(row.get("leverage") or 1.0))
    except (TypeError, ValueError):
        return 10.0 if status != "hit_stop_loss" else -10.0
    if entry <= 0:
        return 10.0 if status != "hit_stop_loss" else -10.0

    if status == "hit_target_2" and t2 > 0:
        gross_pct = abs(t2 - entry) / entry * 100.0
    elif status == "hit_target_1" and t1 > 0:
        gross_pct = abs(t1 - entry) / entry * 100.0
    elif status == "hit_stop_loss" and sl > 0:
        gross_pct = abs(sl - entry) / entry * 100.0
    else:
        gross_pct = 5.0

    leveraged = gross_pct * lev
    if status == "hit_stop_loss":
        return -float(min(30.0, leveraged))
    return float(min(60.0, leveraged))


def train_policy_from_supabase(
    supabase: Any,
    *,
    lookback_days: int = 90,
    min_samples: int = 20,
    cold_start_threshold: int = 60,
    cold_start_weight: float = 0.5,
    save_path: Optional[Path] = None,
    agent_type: Optional[str] = None,
) -> Optional[SignalPolicyBundle]:
    """Train policy with leverage-weighted labels and cold-start support.

    - Sample weights = |leveraged_pnl_pct| (winners with bigger gains and losers with bigger
      losses get more influence than barely-resolved trades).
    - When n < cold_start_threshold (default 60), apply cold_start_weight (0.5) discount so
      the freshly trained bundle is less aggressive until enough data accumulates.
    - min_samples lowered to 20 (was 60) so bootstrap can begin sooner.
    """
    rows = fetch_training_rows_from_supabase(supabase, lookback_days=lookback_days)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    w_list: List[float] = []
    observed_list: List[datetime] = []
    known_list: List[datetime] = []

    for index, row in enumerate(rows):
        vec = build_policy_feature_vector_from_signal_row(row)
        if vec is None:
            continue
        observed_at = parse_utc_datetime(row.get("created_at"))
        if observed_at is None:
            logger.debug("Policy training row %s has no valid created_at; excluding it", index)
            continue
        st = str(row.get("status") or "").lower()
        if st in {"hit_target_1", "hit_target_2"}:
            y_list.append(1)
        elif st == "hit_stop_loss":
            y_list.append(0)
        else:
            continue
        X_list.append(vec.reshape(-1))
        # Leverage-weighted importance — winners with larger leveraged gains and losers with
        # larger leveraged losses dominate over near-breakeven outcomes.
        leveraged_pnl = _compute_leveraged_pnl_pct(row)
        w_list.append(max(0.5, min(20.0, abs(leveraged_pnl))))
        observed_list.append(observed_at)
        known_list.append(outcome_known_at(
            observed_at,
            row.get("expires_at"),
            fallback_purge_hours=24.0,
        ))

    if len(y_list) < min_samples:
        logger.warning(
            "Policy training skipped: only %s resolved samples (min %s)",
            len(y_list),
            min_samples,
        )
        return None

    chronological_order = np.argsort(np.asarray(
        [timestamp.timestamp() for timestamp in observed_list],
        dtype=float,
    ))
    X = np.vstack(X_list)[chronological_order]
    y = np.array(y_list, dtype=np.int32)[chronological_order]
    weights = np.array(w_list, dtype=np.float32)[chronological_order]
    observed_list = [observed_list[index] for index in chronological_order]
    known_list = [known_list[index] for index in chronological_order]

    if len(y_list) < cold_start_threshold:
        weights = weights * float(cold_start_weight)
        logger.info(
            "Policy cold-start: n=%s < %s — applying %.2fx weight discount",
            len(y_list),
            cold_start_threshold,
            cold_start_weight,
        )

    validation_metrics = _walk_forward_policy_validation(
        X,
        y,
        weights,
        observed_list,
        known_list,
    )
    if validation_metrics.get("status") != "passed":
        logger.warning(
            "Policy candidate not promoted: temporal validation status=%s samples=%s brier=%s baseline=%s",
            validation_metrics.get("status"),
            validation_metrics.get("validation_samples", 0),
            validation_metrics.get("brier_score"),
            validation_metrics.get("baseline_brier_score"),
        )
        return None

    pipeline = train_policy_bundle(X, y, sample_weight=weights)
    win_rate = float(np.mean(y))
    out_path = save_path or default_policy_bundle_path()
    save_policy_bundle(out_path, pipeline, len(y_list), win_rate, validation_metrics,
                       db=supabase, agent_type=agent_type)
    logger.info(
        "Policy trained: n=%s win_rate=%.3f mean_weight=%.2f",
        len(y_list),
        win_rate,
        float(np.mean(weights)),
    )
    return SignalPolicyBundle(
        pipeline=pipeline,
        feature_version=POLICY_FEATURE_VERSION,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_samples=len(y_list),
        win_rate=win_rate,
        validation_metrics=validation_metrics,
        path=out_path,
    )
