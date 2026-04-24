# Preset Profile Customization

**Date:** 2026-04-01
**Status:** Draft
**Decision IDs:** DEC-PROFILE-001 (preset system), DEC-PROFILE-002 (curated symbols)

## Context

CerebrumCoin's trading parameters are fully configurable via TOML, but strategy
selection, capital allocation, signal weights, and per-strategy risk overrides
are hardcoded in Python. This makes the system inaccessible to non-technical
users who want to adjust their trading style and risk tolerance without reading
source code.

**Goal:** A non-technical user should be able to configure CerebrumCoin with two
lines of TOML: a risk profile name and their preferred symbol pairs.

## Design

### Two-Layer Config

1. **Simple layer (profiles):** A `[profile]` section in TOML with `name` and
   `symbols`. The profile name maps to a preset bundle that controls strategies,
   capital, and risk parameters. Non-technical users only touch this.

2. **Power-user layer (overrides):** Existing `[risk]`, `[signals]`, `[regime]`,
   and `[strategy.X]` TOML sections override the preset values. Explicit TOML
   keys always win over preset defaults.

**Override detection:** Pydantic populates all fields with defaults, so we
cannot simply check "is this field set." Instead, `load_profile()` compares
each `[risk]` field against `RiskConfig()` defaults. If a field differs from
the default AND the TOML file contains that key, it's a user override. In
practice, we parse the raw TOML dict separately and check which keys are
explicitly present in `[risk]` — only those keys override the preset.

**Backward compatibility:** If `[profile]` is absent, the system falls through
to the current hardcoded strategy registration in `main.py`. Nothing breaks.

### User-Facing Config

```toml
[profile]
name = "moderate"                             # conservative | moderate | aggressive
symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]   # subset of profile's allowed pairs

# Optional power-user overrides:
# [risk]
# stop_loss_percent = "1.2"
```

### Profile Definitions

| Parameter | Conservative | Moderate | Aggressive |
|---|---|---|---|
| **Strategies** | mean_reversion | mean_reversion + range_trading | mean_reversion + range_trading + momentum |
| **Allowed Symbols** | BTC/USD, ETH/USD | BTC/USD, ETH/USD, SOL/USD | BTC/USD, ETH/USD, SOL/USD, DOGE/USD |
| **Capital Split** | 100% MR | 60% MR / 40% RT | 40% MR / 30% RT / 30% MOM |
| **Stop-Loss** | 0.8% | 1.0% | 1.5% |
| **Take-Profit** | 2.0% | 3.0% | 4.0% |
| **Position Size** | 3% | 5% | 7% |
| **Cooldown** | 2400s | 1800s | 900s |
| **Min Signal Strength** | 0.7 | 0.65 | 0.55 |
| **Max Drawdown** | 3% | 5% | 8% |

Rationale:
- **Conservative** uses only mean_reversion (highest proven WR at 30.6%) on
  blue-chip pairs. Tight stops and high signal bar minimize losses.
- **Moderate** matches the current production config (Session 20). Two
  strategies with differentiated signal sources.
- **Aggressive** re-enables momentum (disabled in DEC-TUNE-008 due to signal
  cannibalization with other strategies — but with only 3 strategies the overlap
  is reduced). Wider stops, more volatile pairs, faster cooldowns.

### Curated Symbol Pools (DEC-PROFILE-002)

Users select symbols from a curated list defined per profile. They can remove
symbols but cannot add symbols outside the pool. This prevents picking illiquid
or unsupported pairs.

- **Conservative:** `["BTC/USD", "ETH/USD"]`
- **Moderate:** `["BTC/USD", "ETH/USD", "SOL/USD"]`
- **Aggressive:** `["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]`

If `symbols` is omitted, the profile's full pool is used as default.

### Per-Strategy Symbol Routing

When a profile activates multiple strategies, all strategies receive the same
symbol list from `[profile] symbols`. Per-strategy symbol overrides (e.g.,
momentum excluding BTC as in DEC-TUNE-006) are preserved as hardcoded
exclusions within the strategy config definitions. The profile symbols set the
upper bound; individual strategies may filter further based on their own logic.

## Implementation

### New File: `cerebrum/profiles.py` (~120 lines)

```python
@dataclass(frozen=True)
class ProfilePreset:
    """Immutable preset bundle for a risk profile."""
    name: str
    description: str
    strategies: list[str]           # strategy names to activate
    allowed_symbols: list[str]      # curated symbol pool
    capital_splits: dict[str, Decimal]  # strategy_name -> fraction (must sum to 1.0)
    risk_defaults: dict[str, Any]   # keys match RiskConfig field names
    per_strategy_risk: dict[str, dict[str, Any]]  # strategy_name -> risk_overrides

PROFILES: dict[str, ProfilePreset] = {
    "conservative": ProfilePreset(...),
    "moderate": ProfilePreset(...),
    "aggressive": ProfilePreset(...),
}

@dataclass(frozen=True)
class ResolvedProfile:
    """Fully resolved profile ready for pipeline construction."""
    name: str
    strategies: list[str]
    symbols: list[str]
    total_capital: Decimal
    capital_per_strategy: dict[str, Decimal]
    risk_overrides: dict[str, Any]      # merged: preset + TOML overrides
    per_strategy_risk: dict[str, dict[str, Any]]

def load_profile(config: Config, raw_toml: dict | None = None) -> ResolvedProfile | None:
    """
    Resolve profile from config. Returns None if no [profile] section.

    Args:
        config: Parsed Config object
        raw_toml: Raw TOML dict (pre-pydantic) for detecting explicit overrides.
                  If None, no overrides are applied (pure preset).

    Steps:
    1. Look up PROFILES[config.profile.name]
    2. Validate config.profile.symbols subset of preset.allowed_symbols
    3. Calculate capital_per_strategy from total capital * splits
    4. Detect explicit [risk] keys in raw_toml, merge those over preset defaults
    5. Return ResolvedProfile
    """
```

### Modified: `cerebrum/core/config.py` (~15 lines)

Add a new pydantic model and field:

```python
class ProfileConfig(BaseSettings):
    """User-facing risk profile selection."""
    name: str = ""              # empty = no profile, use legacy path
    symbols: list[str] = []     # empty = use profile's full default pool

    model_config = SettingsConfigDict(
        env_prefix="PROFILE_",
        extra="ignore",
    )

class Config(BaseSettings):
    # ... existing fields ...
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
```

### Modified: `cerebrum/main.py` `_setup_multi_strategy()` (~40 lines)

```python
async def _setup_multi_strategy(self) -> None:
    from cerebrum.profiles import load_profile

    resolved = load_profile(self.config)

    if resolved is not None:
        # Profile-driven path: register strategies from resolved profile
        for strategy_name in resolved.strategies:
            base_config = STRATEGY_CONFIGS[strategy_name]  # existing Python configs
            modified = dataclasses.replace(
                base_config,
                symbols=resolved.symbols,  # profile symbols (strategy may filter further)
                initial_balance=resolved.capital_per_strategy[strategy_name],
                risk_overrides={**base_config.risk_overrides, **resolved.per_strategy_risk.get(strategy_name, {})},
            )
            self.strategy_registry.register(modified)
    else:
        # Legacy path: current hardcoded registration (backward compat)
        self.strategy_registry.register(MEAN_REVERSION_CONFIG)
        self.strategy_registry.register(RANGE_TRADING_CONFIG)
```

### Modified: Strategy Config Files (~5 lines each)

No structural changes needed. The existing `StrategyConfig` is a frozen
dataclass — `dataclasses.replace()` creates a new instance with overridden
fields. The Python-defined configs (MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG,
RANGE_TRADING_CONFIG) serve as base templates that the profile modifies.

Strategy-specific logic (e.g., momentum's BTC exclusion via DEC-TUNE-006,
range_trading's `signal_source_filter`, `exit_monitor_factory`) is preserved
in the base configs. The profile only overrides `symbols`,
`initial_balance`, and `risk_overrides`.

### Strategy Config Lookup

A `STRATEGY_CONFIGS` dict in `profiles.py` maps strategy names to their
base Python configs:

```python
from cerebrum.strategies.momentum import MOMENTUM_CONFIG
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG

STRATEGY_CONFIGS: dict[str, StrategyConfig] = {
    "momentum": MOMENTUM_CONFIG,
    "mean_reversion": MEAN_REVERSION_CONFIG,
    "range_trading": RANGE_TRADING_CONFIG,
}
```

### Per-Strategy Symbol Filtering

When profile sets `symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]`:
- `mean_reversion` receives all three (no exclusions)
- `range_trading` receives all three (no exclusions)
- `momentum` receives `["ETH/USD", "SOL/USD"]` — BTC/USD excluded per
  DEC-TUNE-006 (hardcoded in MOMENTUM_CONFIG base template)

Implementation: intersect profile symbols with the base config's symbols list.
If the base config uses the StrategyConfig default (`["BTC/USD", "ETH/USD"]`),
replace entirely with profile symbols. If the base config has a custom list
(e.g., momentum's `["ETH/USD", "SOL/USD", "DOGE/USD"]`), intersect:
`final = [s for s in profile_symbols if s in base_config.symbols]`. This
preserves DEC-TUNE-006 (momentum excludes BTC) without special-casing.

## Validation

| Condition | Error |
|---|---|
| Unknown profile name | `ConfigError: Unknown profile 'foo'. Available: conservative, moderate, aggressive` |
| Symbol not in pool | `ConfigError: Symbol 'SHIB/USD' not allowed in 'conservative' profile. Allowed: BTC/USD, ETH/USD` |
| Empty symbols after filtering | `ConfigError: No symbols configured. Add at least one symbol to [profile] symbols` |
| Profile name empty string | No error — falls through to legacy path |

## Testing

### Unit Tests (`tests/unit/test_profiles.py`)

1. **test_load_conservative_profile** — resolves to mean_reversion only, BTC+ETH
   symbols, 100% capital allocation, conservative risk params
2. **test_load_moderate_profile** — resolves to MR+RT, correct capital split
   (60/40), moderate risk params
3. **test_load_aggressive_profile** — resolves to MR+RT+MOM, correct capital
   split (40/30/30), aggressive risk params
4. **test_symbol_validation_rejects_unknown** — SHIB/USD in conservative raises
   ConfigError
5. **test_symbol_subset_allowed** — picking only BTC/USD from moderate pool works
6. **test_empty_symbols_uses_profile_default** — omitting symbols uses full pool
7. **test_risk_override_merges_on_preset** — explicit `[risk] stop_loss_percent`
   in TOML overrides the preset value
8. **test_no_profile_returns_none** — empty profile name returns None (legacy)
9. **test_per_strategy_symbol_filtering** — momentum excludes BTC even when
   profile includes it

### Integration Test

10. **test_profile_driven_multi_strategy_boot** — load a TOML with
    `[profile] name = "conservative"`, construct CerebrumApp, verify only
    mean_reversion is registered in the strategy_registry with correct params

## Files Changed

| File | Change | Lines |
|---|---|---|
| `cerebrum/profiles.py` | **New** — ProfilePreset, PROFILES, ResolvedProfile, load_profile() | ~120 |
| `cerebrum/core/config.py` | Add ProfileConfig model + field on Config | ~15 |
| `cerebrum/main.py` | Profile-driven strategy registration in _setup_multi_strategy() | ~40 |
| `config/paper.toml` | Add commented-out `[profile]` section with documentation | ~10 |
| `tests/unit/test_profiles.py` | **New** — 10 tests covering all profiles, validation, merging | ~200 |

**Total:** ~385 lines across 5 files (2 new, 3 modified)

## Verification

1. Run `pytest tests/unit/test_profiles.py -v` — all 10 tests pass
2. Run `pytest` full suite — no regressions (expect 703+ pass)
3. Manual: add `[profile] name = "moderate"` to paper.toml, start app in paper
   mode, verify log shows mean_reversion + range_trading registered with correct
   capital and symbols
4. Manual: change to `name = "conservative"`, restart, verify only mean_reversion
   registered
5. Manual: add invalid symbol, verify clear error message on startup
