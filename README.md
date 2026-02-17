# CerebrumCoin

Autonomous adaptive AI trading agent with hybrid ML+LLM intelligence.

## Phase 1: Foundation

Event-driven architecture with live Kraken data integration and paper trading.

### Installation

```bash
pip install -e .
```

### Running

```bash
# Paper trading mode (default)
python -m cerebrum --mode paper

# With specific config
python -m cerebrum --config config/paper.toml
```

### Testing

```bash
pytest
```

### Architecture

```
[Data Sources] → [Event Bus] → [Signal Pipeline] → [Aggregator] → [Risk Manager] → [Executor]
       ↑                                                                                   |
       └──────────────────── [Closed-Loop Learner] ←───────────────────────────────────────┘
```

See `MASTER_PLAN.md` for full roadmap and decision log.
