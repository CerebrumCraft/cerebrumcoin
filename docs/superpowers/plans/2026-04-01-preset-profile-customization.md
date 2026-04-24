# Preset Profile Customization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Non-technical users configure their trading style with two TOML lines: a risk profile name and symbol selection.

**Architecture:** A new `cerebrum/profiles.py` module defines three preset bundles (conservative/moderate/aggressive). `Config.from_toml()` returns raw TOML alongside the parsed Config so `load_profile()` can detect explicit user overrides. `main.py._setup_multi_strategy()` checks for a profile and uses it to drive strategy registration, replacing hardcoded strategy selection.

**Tech Stack:** Python dataclasses, pydantic-settings, tomllib, pytest

**Spec:** `docs/superpowers/specs/2026-04-01-preset-profile-customization-design.md`

---

## File Structure

| File | Role |
|------|------|
| `cerebrum/profiles.py` | **New.** ProfilePreset, PROFILES dict, ResolvedProfile, `load_profile()` |
| `cerebrum/core/config.py` | **Modify.** Add `ProfileConfig` model, add `profile` field to `Config`, modify `from_toml()` to return raw TOML |
| `cerebrum/main.py` | **Modify.** Profile-driven strategy registration in `_setup_multi_strategy()` |
| `config/paper.toml` | **Modify.** Add commented-out `[profile]` section with documentation |
| `tests/unit/test_profiles.py` | **New.** 10 unit tests for profile loading, validation, merging |

---

### Task 1: Add ProfileConfig to config.py

**Files:**
- Modify: `cerebrum/core/config.py:452-486`
- Test: `tests/unit/test_config.py` (existing)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_config.py`:

```python
def test_config_profile_defaults():
    """ProfileConfig has empty defaults when not specified."""
    config = Config()
    assert config.profile.name == ""
    assert config.profile.symbols == []


def test_config_profile_from_toml(tmp_path):
    """Profile section is parsed from TOML."""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text('''
[profile]
name = "moderate"
symbols = ["BTC/USD", "ETH/USD"]
''')
    config, raw = Config.from_toml(toml_file)
    assert config.profile.name == "moderate"
    assert config.profile.symbols == ["BTC/USD", "ETH/USD"]
    assert "profile" in raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_config_profile_defaults tests/unit/test_config.py::test_config_profile_from_toml -v`
Expected: FAIL — `Config` has no `profile` attribute; `from_toml` returns `Config` not tuple.

- [ ] **Step 3: Add ProfileConfig model and modify from_toml**

In `cerebrum/core/config.py`, add `ProfileConfig` class before the `Config` class (around line 450):

```python
class ProfileConfig(BaseSettings):
    """User-facing risk profile selection."""
    name: str = ""
    symbols: list[str] = Field(default_factory=list)

    model_config = SettingsConfigDict(
        env_prefix="PROFILE_",
        extra="ignore",
    )
```

Add `profile` field to `Config` class (after `alpaca` field, around line 464):

```python
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
```

Modify `from_toml()` to return the raw TOML dict alongside the Config:

```python
    @classmethod
    def from_toml(cls, toml_path: Path) -> tuple["Config", dict]:
        """Load configuration from a TOML file, with env var overrides.

        Returns:
            Tuple of (Config, raw_toml_dict). The raw dict is used by
            load_profile() to detect explicit user overrides vs defaults.
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        if not toml_path.exists():
            return cls(), {}

        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)

        return cls(**toml_data), toml_data
```

- [ ] **Step 4: Fix all callers of Config.from_toml()**

`from_toml()` now returns a tuple. Find and update all call sites. The main one is in `cerebrum/main.py` at the top-level `main()` function (search for `Config.from_toml`). Update each call to unpack:

```python
config, raw_toml = Config.from_toml(args.config)
```

Store `raw_toml` on the `CerebrumCoin` instance so `_setup_multi_strategy()` can access it:

In `CerebrumCoin.__init__()`, add a parameter:

```python
def __init__(self, config: Config, raw_toml: dict | None = None) -> None:
    ...
    self._raw_toml = raw_toml or {}
```

And at the call site in `main()`, pass it through:

```python
app = CerebrumCoin(config, raw_toml=raw_toml)
```

Also check `tests/unit/test_config.py` and any other test that calls `Config.from_toml()` — update them to unpack the tuple.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: All pass including the two new tests.

- [ ] **Step 6: Run full test suite to check for from_toml breakage**

Run: `pytest --timeout=120 -x -q`
Expected: All 703+ pass. If any fail, they're likely calling `from_toml()` and expecting a Config, not a tuple — fix those callers.

- [ ] **Step 7: Commit**

```bash
git add cerebrum/core/config.py cerebrum/main.py tests/unit/test_config.py
git commit -m "feat: add ProfileConfig to config and return raw TOML from from_toml (DEC-PROFILE-001)"
```

---

### Task 2: Create profiles.py with preset definitions

**Files:**
- Create: `cerebrum/profiles.py`
- Test: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_profiles.py`:

```python
"""Tests for preset profile system (DEC-PROFILE-001, DEC-PROFILE-002)."""

from decimal import Decimal

import pytest

from cerebrum.core.config import Config, ProfileConfig
from cerebrum.profiles import (
    PROFILES,
    ConfigError,
    ResolvedProfile,
    load_profile,
)


class TestProfileDefinitions:
    """Verify preset bundles are internally consistent."""

    def test_all_profiles_capital_splits_sum_to_one(self):
        for name, preset in PROFILES.items():
            total = sum(preset.capital_splits.values())
            assert total == Decimal("1.0"), (
                f"Profile '{name}' capital splits sum to {total}, expected 1.0"
            )

    def test_all_profiles_strategies_match_capital_keys(self):
        for name, preset in PROFILES.items():
            assert set(preset.strategies) == set(preset.capital_splits.keys()), (
                f"Profile '{name}' strategy list doesn't match capital_splits keys"
            )


class TestLoadConservative:
    def test_resolves_mean_reversion_only(self):
        config = Config(
            profile=ProfileConfig(name="conservative", symbols=["BTC/USD"]),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result is not None
        assert result.strategies == ["mean_reversion"]

    def test_full_capital_to_mean_reversion(self):
        config = Config(
            profile=ProfileConfig(name="conservative"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.capital_per_strategy["mean_reversion"] == Decimal("10000.0")

    def test_conservative_risk_params(self):
        config = Config(
            profile=ProfileConfig(name="conservative"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.risk_overrides["stop_loss_percent"] == "0.8"
        assert result.risk_overrides["position_size_percent"] == "3.0"


class TestLoadModerate:
    def test_resolves_two_strategies(self):
        config = Config(
            profile=ProfileConfig(name="moderate"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.strategies == ["mean_reversion", "range_trading"]

    def test_capital_split_60_40(self):
        config = Config(
            profile=ProfileConfig(name="moderate"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.capital_per_strategy["mean_reversion"] == Decimal("6000.0")
        assert result.capital_per_strategy["range_trading"] == Decimal("4000.0")


class TestLoadAggressive:
    def test_resolves_three_strategies(self):
        config = Config(
            profile=ProfileConfig(name="aggressive"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.strategies == ["mean_reversion", "range_trading", "momentum"]

    def test_capital_split_40_30_30(self):
        config = Config(
            profile=ProfileConfig(name="aggressive"),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.capital_per_strategy["mean_reversion"] == Decimal("4000.0")
        assert result.capital_per_strategy["range_trading"] == Decimal("3000.0")
        assert result.capital_per_strategy["momentum"] == Decimal("3000.0")


class TestSymbolValidation:
    def test_rejects_symbol_outside_pool(self):
        config = Config(
            profile=ProfileConfig(
                name="conservative", symbols=["SHIB/USD"]
            ),
            paper={"initial_balance_usd": "10000.0"},
        )
        with pytest.raises(ConfigError, match="not allowed.*conservative"):
            load_profile(config)

    def test_subset_of_pool_allowed(self):
        config = Config(
            profile=ProfileConfig(
                name="moderate", symbols=["BTC/USD"]
            ),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.symbols == ["BTC/USD"]

    def test_empty_symbols_uses_full_pool(self):
        config = Config(
            profile=ProfileConfig(name="moderate", symbols=[]),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        assert result.symbols == ["BTC/USD", "ETH/USD", "SOL/USD"]


class TestOverrideMerging:
    def test_risk_override_merges_on_preset(self):
        config = Config(
            profile=ProfileConfig(name="conservative"),
            paper={"initial_balance_usd": "10000.0"},
        )
        raw_toml = {"risk": {"stop_loss_percent": "1.2"}}
        result = load_profile(config, raw_toml=raw_toml)
        assert result.risk_overrides["stop_loss_percent"] == "1.2"
        # Other preset values preserved
        assert result.risk_overrides["position_size_percent"] == "3.0"


class TestNoProfile:
    def test_empty_name_returns_none(self):
        config = Config(profile=ProfileConfig(name=""))
        result = load_profile(config)
        assert result is None

    def test_no_profile_section_returns_none(self):
        config = Config()
        result = load_profile(config)
        assert result is None


class TestUnknownProfile:
    def test_unknown_name_raises_config_error(self):
        config = Config(
            profile=ProfileConfig(name="yolo"),
            paper={"initial_balance_usd": "10000.0"},
        )
        with pytest.raises(ConfigError, match="Unknown profile 'yolo'"):
            load_profile(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_profiles.py -v`
Expected: FAIL — `cerebrum.profiles` does not exist.

- [ ] **Step 3: Create cerebrum/profiles.py**

Create `cerebrum/profiles.py`:

```python
"""
Preset profile system for non-technical users.

Maps a profile name (conservative/moderate/aggressive) to a complete bundle
of strategy selection, capital allocation, symbol pool, and risk parameters.
Power users can override individual risk parameters via explicit TOML keys.

@decision DEC-PROFILE-001
@title Preset-based strategy customization via [profile] TOML section
@status accepted
@rationale Non-technical users need a simple config surface (2 lines of TOML)
to set trading style and risk tolerance. Presets bundle strategy selection,
capital splits, and risk params. Power users override via explicit [risk] keys.
Backward-compatible: no [profile] section falls through to legacy hardcoded
strategy registration in main.py.

@decision DEC-PROFILE-002
@title Curated symbol pools per profile
@status accepted
@rationale Prevents non-technical users from picking illiquid or unsupported
pairs. Each profile defines an allowed symbol pool. Users can remove symbols
but cannot add symbols outside the pool.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from cerebrum.core.config import Config


class ConfigError(Exception):
    """Raised when profile configuration is invalid."""


@dataclass(frozen=True)
class ProfilePreset:
    """Immutable preset bundle for a risk profile."""
    name: str
    description: str
    strategies: list[str]
    allowed_symbols: list[str]
    capital_splits: dict[str, Decimal]
    risk_defaults: dict[str, Any]
    per_strategy_risk: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ResolvedProfile:
    """Fully resolved profile ready for pipeline construction."""
    name: str
    strategies: list[str]
    symbols: list[str]
    total_capital: Decimal
    capital_per_strategy: dict[str, Decimal]
    risk_overrides: dict[str, Any]
    per_strategy_risk: dict[str, dict[str, Any]]


PROFILES: dict[str, ProfilePreset] = {
    "conservative": ProfilePreset(
        name="conservative",
        description="Low risk. Mean reversion only on blue-chip pairs. Tight stops.",
        strategies=["mean_reversion"],
        allowed_symbols=["BTC/USD", "ETH/USD"],
        capital_splits={"mean_reversion": Decimal("1.0")},
        risk_defaults={
            "stop_loss_percent": "0.8",
            "take_profit_percent": "2.0",
            "position_size_percent": "3.0",
            "post_fill_cooldown_seconds": 2400,
            "min_signal_strength": "0.7",
            "max_drawdown_percent": "3.0",
        },
        per_strategy_risk={
            "mean_reversion": {
                "min_signal_strength": "0.6",
                "position_size_percent": "3.0",
                "post_fill_cooldown_seconds": 2400,
            },
        },
    ),
    "moderate": ProfilePreset(
        name="moderate",
        description="Balanced. Mean reversion + range trading. Current production config.",
        strategies=["mean_reversion", "range_trading"],
        allowed_symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        capital_splits={
            "mean_reversion": Decimal("0.6"),
            "range_trading": Decimal("0.4"),
        },
        risk_defaults={
            "stop_loss_percent": "1.0",
            "take_profit_percent": "3.0",
            "position_size_percent": "5.0",
            "post_fill_cooldown_seconds": 1800,
            "min_signal_strength": "0.65",
            "max_drawdown_percent": "5.0",
        },
        per_strategy_risk={
            "mean_reversion": {
                "min_signal_strength": "0.5",
                "position_size_percent": "5.0",
                "post_fill_cooldown_seconds": 1800,
            },
            "range_trading": {
                "min_signal_strength": "0.3",
                "position_size_percent": "2.0",
                "post_fill_cooldown_seconds": 900,
            },
        },
    ),
    "aggressive": ProfilePreset(
        name="aggressive",
        description="High risk. Three strategies including momentum. Wider stops, more pairs.",
        strategies=["mean_reversion", "range_trading", "momentum"],
        allowed_symbols=["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"],
        capital_splits={
            "mean_reversion": Decimal("0.4"),
            "range_trading": Decimal("0.3"),
            "momentum": Decimal("0.3"),
        },
        risk_defaults={
            "stop_loss_percent": "1.5",
            "take_profit_percent": "4.0",
            "position_size_percent": "7.0",
            "post_fill_cooldown_seconds": 900,
            "min_signal_strength": "0.55",
            "max_drawdown_percent": "8.0",
        },
        per_strategy_risk={
            "mean_reversion": {
                "min_signal_strength": "0.45",
                "position_size_percent": "7.0",
                "post_fill_cooldown_seconds": 900,
            },
            "range_trading": {
                "min_signal_strength": "0.25",
                "position_size_percent": "4.0",
                "post_fill_cooldown_seconds": 600,
            },
            "momentum": {
                "min_signal_strength": "0.55",
                "position_size_percent": "7.0",
                "post_fill_cooldown_seconds": 900,
            },
        },
    ),
}


def load_profile(
    config: Config, raw_toml: dict | None = None
) -> ResolvedProfile | None:
    """
    Resolve a profile from config. Returns None if no profile is set.

    Args:
        config: Parsed Config object with profile.name and profile.symbols.
        raw_toml: Raw TOML dict (pre-pydantic) for detecting explicit [risk]
                  overrides. Only keys present in raw_toml["risk"] override
                  the preset defaults. If None, pure preset values are used.

    Returns:
        ResolvedProfile if a profile name is set, None otherwise.

    Raises:
        ConfigError: Unknown profile name, invalid symbol, or empty symbol list.
    """
    profile_name = config.profile.name
    if not profile_name:
        return None

    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ConfigError(
            f"Unknown profile '{profile_name}'. Available: {available}"
        )

    preset = PROFILES[profile_name]

    # --- Resolve symbols ---
    user_symbols = config.profile.symbols
    if not user_symbols:
        symbols = list(preset.allowed_symbols)
    else:
        for sym in user_symbols:
            if sym not in preset.allowed_symbols:
                allowed = ", ".join(preset.allowed_symbols)
                raise ConfigError(
                    f"Symbol '{sym}' not allowed in '{profile_name}' profile. "
                    f"Allowed: {allowed}"
                )
        symbols = list(user_symbols)

    # --- Calculate capital per strategy ---
    total_capital = config.paper.initial_balance_usd
    capital_per_strategy = {
        name: total_capital * split
        for name, split in preset.capital_splits.items()
    }

    # --- Merge risk overrides: preset defaults + explicit TOML keys ---
    risk_overrides = dict(preset.risk_defaults)
    if raw_toml and "risk" in raw_toml:
        explicit_risk = raw_toml["risk"]
        for key, value in explicit_risk.items():
            risk_overrides[key] = value

    return ResolvedProfile(
        name=profile_name,
        strategies=list(preset.strategies),
        symbols=symbols,
        total_capital=total_capital,
        capital_per_strategy=capital_per_strategy,
        risk_overrides=risk_overrides,
        per_strategy_risk=dict(preset.per_strategy_risk),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_profiles.py -v`
Expected: All 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add cerebrum/profiles.py tests/unit/test_profiles.py
git commit -m "feat: add preset profile system with conservative/moderate/aggressive bundles (DEC-PROFILE-001)"
```

---

### Task 3: Wire profiles into main.py strategy registration

**Files:**
- Modify: `cerebrum/main.py:434-522`
- Test: `tests/unit/test_profiles.py` (add integration test)

- [ ] **Step 1: Write the failing integration test**

Add to `tests/unit/test_profiles.py`:

```python
class TestPerStrategySymbolFiltering:
    def test_momentum_excludes_btc_even_when_profile_includes_it(self):
        """Momentum base config excludes BTC (DEC-TUNE-006). Profile symbols
        are intersected with base config symbols, so BTC is filtered out."""
        from cerebrum.profiles import get_strategy_configs

        config = Config(
            profile=ProfileConfig(
                name="aggressive",
                symbols=["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"],
            ),
            paper={"initial_balance_usd": "10000.0"},
        )
        result = load_profile(config)
        configs = get_strategy_configs(result)
        momentum_cfg = configs["momentum"]
        assert "BTC/USD" not in momentum_cfg.symbols
        assert "ETH/USD" in momentum_cfg.symbols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_profiles.py::TestPerStrategySymbolFiltering -v`
Expected: FAIL — `get_strategy_configs` does not exist.

- [ ] **Step 3: Add get_strategy_configs() to profiles.py**

Add to `cerebrum/profiles.py` at the bottom:

```python
import dataclasses

from cerebrum.strategies.base import StrategyConfig

# Lazy imports to avoid circular dependency — these are only needed
# when actually building strategy configs from a resolved profile.
_STRATEGY_BASE_CONFIGS: dict[str, StrategyConfig] | None = None


def _get_base_configs() -> dict[str, StrategyConfig]:
    """Lazy-load strategy base configs to avoid circular imports."""
    global _STRATEGY_BASE_CONFIGS
    if _STRATEGY_BASE_CONFIGS is None:
        from cerebrum.strategies.momentum import MOMENTUM_CONFIG
        from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
        from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG

        _STRATEGY_BASE_CONFIGS = {
            "momentum": MOMENTUM_CONFIG,
            "mean_reversion": MEAN_REVERSION_CONFIG,
            "range_trading": RANGE_TRADING_CONFIG,
        }
    return _STRATEGY_BASE_CONFIGS


def get_strategy_configs(
    resolved: ResolvedProfile,
) -> dict[str, StrategyConfig]:
    """
    Build StrategyConfig instances from a resolved profile.

    For each strategy in the profile:
    1. Start from the base Python config (preserves signal_source_filter,
       exit_monitor_factory, aggregator_weights, etc.)
    2. Override symbols: intersect profile symbols with base config symbols.
       If base uses the default symbols, use all profile symbols.
    3. Override initial_balance from profile capital allocation.
    4. Merge risk_overrides: base config overrides + profile per-strategy overrides.

    Returns:
        Dict mapping strategy name to modified StrategyConfig.
    """
    base_configs = _get_base_configs()
    default_symbols = StrategyConfig().symbols  # ["BTC/USD", "ETH/USD"]
    result = {}

    for strategy_name in resolved.strategies:
        base = base_configs[strategy_name]

        # Symbol filtering: intersect profile symbols with base config's list.
        # If base uses default symbols, use all profile symbols instead.
        if base.symbols == default_symbols:
            final_symbols = list(resolved.symbols)
        else:
            final_symbols = [s for s in resolved.symbols if s in base.symbols]

        # Merge risk overrides: base strategy overrides + profile per-strategy
        merged_risk = {
            **base.risk_overrides,
            **resolved.per_strategy_risk.get(strategy_name, {}),
        }

        result[strategy_name] = dataclasses.replace(
            base,
            symbols=final_symbols,
            initial_balance=resolved.capital_per_strategy[strategy_name],
            risk_overrides=merged_risk,
        )

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_profiles.py::TestPerStrategySymbolFiltering -v`
Expected: PASS.

- [ ] **Step 5: Wire into main.py _setup_multi_strategy()**

In `cerebrum/main.py`, modify `_setup_multi_strategy()` (around line 497-519). Replace the hardcoded strategy registration block with:

```python
        # --- StrategyRegistry ---
        self.strategy_registry = StrategyRegistry(bus=self.bus, config=config)

        # --- Profile-driven or legacy strategy registration ---
        from cerebrum.profiles import load_profile, get_strategy_configs

        resolved = load_profile(config, raw_toml=self._raw_toml)

        if resolved is not None:
            # Profile path: register strategies from resolved profile
            strategy_configs = get_strategy_configs(resolved)
            for _name, strategy_cfg in strategy_configs.items():
                self.strategy_registry.register(strategy_cfg)
            self._log.info(
                "profile_strategies_registered",
                profile=resolved.name,
                strategies=resolved.strategies,
                symbols=resolved.symbols,
            )
        else:
            # Legacy path: hardcoded registration (no [profile] in TOML)
            # @decision DEC-TUNE-008
            # @title Disable momentum, breakout, news_driven — signal cannibalization
            # @status accepted
            # @rationale See original annotation for full rationale.
            # self.strategy_registry.register(MOMENTUM_CONFIG)
            self.strategy_registry.register(MEAN_REVERSION_CONFIG)
            # self.strategy_registry.register(BREAKOUT_CONFIG)
            self.strategy_registry.register(RANGE_TRADING_CONFIG)
            # self.strategy_registry.register(SWING_TRADING_CONFIG)
            # self.strategy_registry.register(NEWS_DRIVEN_CONFIG)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest --timeout=120 -x -q`
Expected: All 703+ pass (plus new profile tests). No regressions — the legacy path is unchanged when no `[profile]` section exists.

- [ ] **Step 7: Commit**

```bash
git add cerebrum/profiles.py cerebrum/main.py tests/unit/test_profiles.py
git commit -m "feat: wire profile-driven strategy registration into main.py"
```

---

### Task 4: Add documented [profile] section to paper.toml

**Files:**
- Modify: `config/paper.toml:82-84`

- [ ] **Step 1: Replace the existing stub with documented profile section**

The current `paper.toml` has a minimal `[strategy.range_trading]` stub at line 82. Replace it with a comprehensive profile section:

```toml
# --- Risk Profile (optional) ---
# Set a profile to configure strategies, capital, and risk in one line.
# Available profiles:
#   conservative — mean reversion only, BTC+ETH, tight stops
#   moderate     — mean reversion + range trading, +SOL, balanced risk
#   aggressive   — +momentum strategy, +DOGE, wider stops, faster trading
#
# Symbols must be from the profile's allowed pool. Omit to use all.
# Any [risk] values below will override the profile defaults.
#
# [profile]
# name = "moderate"
# symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
```

Remove the old `[strategy.range_trading]` stub since profile now controls this.

- [ ] **Step 2: Verify TOML is valid**

Run: `python -c "import tomllib; tomllib.load(open('config/paper.toml', 'rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full test suite to ensure no regressions**

Run: `pytest --timeout=120 -x -q`
Expected: All pass. No profile is active (commented out), so legacy path is used.

- [ ] **Step 4: Commit**

```bash
git add config/paper.toml
git commit -m "docs: add documented [profile] section to paper.toml"
```

---

### Task 5: Manual verification

- [ ] **Step 1: Test moderate profile**

Temporarily uncomment the `[profile]` section in `config/paper.toml`:

```toml
[profile]
name = "moderate"
symbols = ["BTC/USD", "ETH/USD"]
```

Run: `python -m cerebrum.main --mode paper --dry-run 2>&1 | head -50`
(If `--dry-run` doesn't exist, start the app and check the first few log lines.)

Expected: Log output showing `profile_strategies_registered profile=moderate strategies=['mean_reversion', 'range_trading'] symbols=['BTC/USD', 'ETH/USD']`

- [ ] **Step 2: Test conservative profile**

Change to `name = "conservative"` and restart.

Expected: Only `mean_reversion` registered.

- [ ] **Step 3: Test invalid symbol**

Change to `symbols = ["SHIB/USD"]` with `name = "conservative"`.

Expected: `ConfigError: Symbol 'SHIB/USD' not allowed in 'conservative' profile. Allowed: BTC/USD, ETH/USD`

- [ ] **Step 4: Revert paper.toml to commented-out state**

Re-comment the `[profile]` section so paper.toml stays in legacy mode for current sessions.

- [ ] **Step 5: Final full test suite**

Run: `pytest --timeout=120 -q`
Expected: All 703+ pass plus new profile tests. No regressions.

- [ ] **Step 6: Commit any test fixes**

If any fixes were needed during verification, commit them:

```bash
git add -u
git commit -m "fix: address issues found during manual profile verification"
```
