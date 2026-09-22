# Kata ⚔️

**A mobile-first autonomous perpetual futures dApp that makes quantitative crypto investing effortless and safe for everyone.**

Kata puts an institutional-grade quantitative hedge fund in your pocket. Trading perpetuals is notoriously hostile for everyday users—over 90% lose capital to emotional trading, liquidation traps, and complex orderbooks. Kata changes this: connect your wallet, choose your risk comfort, and let an autonomous, self-improving pilot manage execution 24/7.

Most "AI trading bots" are simply language models that print reckless opinions. Kata’s real edge is **everything downstream of the signal**: calibrated confidence floors, walk-forward expected-value (EV) vetoes, idempotent order execution, and a 3-tier closed-loop learning engine that measurably shifts parameters as real trade outcomes land.

---

## The Core Differentiator: Downstream Risk Gating

Kata's edge is capital preservation. Before any order is dispatched, it must survive rigorous statistical risk gates:

1. **Calibrated Confidence Floor**: Signals below a statistically backed threshold (e.g. 60%) are discarded before entry.
2. **Expected Value (EV) Veto**: Uses a purged walk-forward logistic policy trained on historical outcomes. If the modeled net expectancy is negative ($P(\text{win}) \cdot \text{gain} - P(\text{loss}) \cdot \text{loss} \le 0$), the signal is hard-vetoed.
3. **The Downstream Veto Feed ("Why I Didn't Trade")**: Rather than hiding rejections, Kata transparently logs every blocked setup with its mathematical justification—saving users from unneeded drawdowns.

---

## 3-Tier Closed-Loop Learning Architecture

Kata does not remain static; the loop is the product:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KATA LEARNING CYCLES                            │
└────────────────────────────────────────────────────────────────────────┘
  Level 1: Real-Time Feedback (Every 30s)
  └── Tracks active positions, MFE/MAE excursions, and dynamic trailing stops.

  Level 2: Pattern Matrix Clustering (Every 15m)
  └── Evaluates 24+ market archetypes (RSI states, volatility regimes, breakout vs mean rev)
      to adjust confidence multipliers and position sizes.

  Level 3: Meta-Learning Optimization (Daily)
  └── Retrains walk-forward policy models and recalibrates global confidence floors.
```

---

## Venue-Agnostic Execution Engine

The agent depends strictly on the `PerpVenue` abstraction, never on a concrete exchange:

```
kata/venues/
├── types.py         # Venue-neutral Quote, Position, OrderRequest, OrderResult
├── base.py          # The PerpVenue abstract contract
├── hyperliquid.py   # Hyperliquid L1 adapter (live delegated execution)
├── drift.py         # Drift Protocol / Solana native adapter
└── registry.py      # Dynamic venue resolution (get_venue)
```

The same signal pipeline, risk engine, and learning loop operate unchanged across venues. Adding a new DEX requires an adapter, not a rewrite.

---

## Web & Mobile Cockpit

Kata includes a high-performance web cockpit built with an obsidian cybernetic aesthetic:
- **Live Telemetry & Pilot Status**: Real-time market regime detection and active confidence floors.
- **Agent Brain (ReAct Stream)**: Live streaming thought feed showing market scanning, thesis synthesis, and order fills.
- **Downstream Veto Feed**: Transparent ledger of bad setups the agent saved you from.
- **Public Verifiable Track Record**: Ledger of closed trades with on-chain verification links.
- **Non-Custodial Allocation Console**: 1-tap deposit, leverage sliders (1x–10x), and emergency pause controls.

---

## Quickstart & Local Development

### 1. Installation
```bash
git clone https://github.com/ayomidefagboyo/kata.git
cd kata
pip install -r requirements.txt
cp .env.example .env   # Configure database, venue, and LLM credentials
```

### 2. Run the Web Cockpit & API
```bash
python start.py web
```
Visit **`http://localhost:8000/`** to view the live Kata Cockpit.

### 3. Run the Full Stack (Web + Background Learning Loop)
```bash
python start.py all
```

### 4. Run via Docker Compose
```bash
docker compose up -d --build
```

---

## Architecture Overview

- **`kata/venues/`**: Exchange-agnostic perp contract and adapters (Hyperliquid L1, Solana Drift).
- **`kata/agents/`**: Autonomous ReAct agent implementations (`Yuki` perp pilot, `AgentManager`).
- **`kata/services/`**: Multi-level learning engine, offline walk-forward policy, and capital allocation.
- **`kata/workers/`**: Continuous background pipelines (real-time learning, market signals scanner).
- **`kata/api/`**: FastAPI REST & WebSocket endpoints (`/ws/{user_id}`).
- **`kata/sql/`**: Consolidated database schema (`schema.sql`) for PostgreSQL / Supabase.
- **`client/`**: Lightweight, high-conviction mobile-first trading cockpit.

---

## License

Proprietary. All rights reserved.
