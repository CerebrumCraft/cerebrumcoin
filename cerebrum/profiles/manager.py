"""
ProfileManager: hot-swappable risk profile application.

Parses named profiles from the [profiles.*] sections of raw TOML config and
applies them to live pipeline components without restarting the process.

Usage::

    config, raw_toml = Config.from_toml(Path("config/paper.toml"))
    manager = ProfileManager(registry, raw_toml, default_profile="moderate")
    manager.apply_profile("conservative")

@decision DEC-PROFILE-002
@title ProfileManager mutates private pipeline attributes for hot-swap
@status accepted
@rationale The pipeline components (ExitMonitor, RiskManager rules, SignalAggregator)
store their thresholds as simple instance attributes (e.g. _stop_loss_pct, _threshold).
There are no public setters because these values were designed to be set once at
construction time. Two approaches were considered:

  Option A — add public setters to each component class.
    Pro: clean API, type-safe.
    Con: every component needs new API surface; Phase 14B dashboard and any future
    caller must know which setter to call per component. Multiplies coupling.

  Option B — ProfileManager mutates private attributes directly (chosen).
    Pro: zero changes to component classes; single choke-point for all overrides;
    easy to extend with new fields by adding one line here.
    Con: relies on private naming convention; breaks if a component renames an attr.
    Mitigation: the attribute names are verified in tests and documented here.

Attribute map (verified against source):
  PositionSizingRule      ._size_percent
  MinSignalStrengthRule   ._min_strength
  PostFillCooldownRule    ._cooldown_seconds
  ExitMonitor             ._stop_loss_pct, ._take_profit_pct, ._max_age_seconds (seconds!),
                          ._adaptive_tp, ._tp_multiplier, ._min_tp_percent
  SignalAggregator        ._threshold

RangeExitMonitor uses structural S/R exits, not the same percentage-based
parameters — it is intentionally excluded from profile application.
"""

from typing import Any

import structlog

from cerebrum.core.config import ProfileConfig
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.rules import MinSignalStrengthRule, PositionSizingRule, PostFillCooldownRule

logger = structlog.get_logger()


class ProfileManager:
    """
    Manages named risk profiles and applies them to live strategy pipelines.

    Profiles are parsed from the ``[profiles.*]`` sections of the raw TOML dict
    returned by ``Config.from_toml()``. Each profile is a ``ProfileConfig`` whose
    ``None`` fields mean "don't override this parameter".

    When ``apply_profile()`` is called, the manager walks every active pipeline
    in the ``StrategyRegistry`` and mutates the relevant component attributes in
    place. The change is immediate — the next market tick will use the new values.

    Thread-safety: CerebrumCoin runs single-threaded asyncio. No locking is
    needed here; apply_profile() is safe to call from a coroutine context.
    """

    def __init__(
        self,
        registry: Any,  # StrategyRegistry — avoid circular import
        raw_toml: dict,
        default_profile: str = "",
    ) -> None:
        """
        Initialize the profile manager.

        Args:
            registry: The live ``StrategyRegistry`` whose pipelines will be
                      mutated when ``apply_profile()`` is called. Must have
                      ``active_strategy_names()``, ``get_risk_manager()``,
                      ``get_exit_monitor()``, and ``get_aggregator()`` methods.
            raw_toml: Full raw TOML dict from ``Config.from_toml()``. Profile
                      sections are parsed from ``raw_toml["profiles"]``.
            default_profile: Profile name to treat as active at startup.
                             Does NOT apply the profile — caller must call
                             ``apply_profile()`` explicitly if desired.
                             Empty string means no profile is active yet.
        """
        self._registry = registry
        self._log = logger.bind(component="profile_manager")

        # Parse profiles from [profiles.*] sections
        self._profiles: dict[str, ProfileConfig] = {}
        profiles_section = raw_toml.get("profiles", {})
        for name, fields in profiles_section.items():
            try:
                profile = ProfileConfig(**fields)
                self._profiles[name] = profile
                self._log.info("profile_loaded", name=name, fields=list(fields.keys()))
            except Exception as exc:
                self._log.warning(
                    "profile_parse_failed",
                    name=name,
                    error=str(exc),
                )

        self._active_profile: str = default_profile
        self._log.info(
            "profile_manager_initialized",
            profiles=list(self._profiles.keys()),
            default_profile=default_profile or "(none)",
        )

    # --- Public API ---

    def list_profiles(self) -> list[str]:
        """Return the names of all available profiles."""
        return list(self._profiles.keys())

    def get_active_profile(self) -> str:
        """Return the name of the currently active profile, or '' if none applied."""
        return self._active_profile

    def get_profile_config(self, name: str) -> ProfileConfig:
        """
        Return the ProfileConfig for a named profile.

        Args:
            name: Profile name (e.g. "conservative").

        Raises:
            ValueError: If the profile name is not found.
        """
        if name not in self._profiles:
            available = list(self._profiles.keys())
            raise ValueError(
                f"Unknown profile '{name}'. Available profiles: {available}"
            )
        return self._profiles[name]

    def apply_profile(self, name: str) -> dict:
        """
        Apply a named profile to all active strategy pipelines.

        Walks every active pipeline and mutates the relevant component
        attributes in place. Changes take effect on the next event cycle —
        no restart required.

        Args:
            name: Profile name to activate (must exist in list_profiles()).

        Returns:
            Dict mapping ``"strategy_name.component.field"`` to new value for
            every override applied. Empty dict if no active pipelines exist.

        Raises:
            ValueError: If profile name is not found.

        Side-effects:
            - Logs a WARNING for each strategy that has open positions when a
              stop-loss tightening is applied (new SL < old SL), because the
              tighter threshold could trigger an immediate exit on the next tick.
        """
        profile = self.get_profile_config(name)  # raises ValueError if missing
        changes: dict = {}

        active_names = self._registry.active_strategy_names()
        if not active_names:
            self._log.warning("apply_profile_no_active_strategies", profile=name)
            self._active_profile = name
            return changes

        for strategy_name in active_names:
            strategy_changes = self._apply_to_pipeline(strategy_name, profile)
            changes.update(strategy_changes)

        self._active_profile = name
        self._log.info(
            "profile_applied",
            profile=name,
            strategies=active_names,
            changes_count=len(changes),
        )
        return changes

    # --- Internal helpers ---

    def _apply_to_pipeline(self, strategy_name: str, profile: ProfileConfig) -> dict:
        """
        Apply profile overrides to a single strategy pipeline.

        Returns a flat dict of ``"strategy.component.field" -> new_value`` for
        every override that was applied (skips None fields).
        """
        changes: dict = {}
        prefix = strategy_name

        risk_manager = self._registry.get_risk_manager(strategy_name)
        exit_monitor = self._registry.get_exit_monitor(strategy_name)
        aggregator = self._registry.get_aggregator(strategy_name)
        portfolio = self._registry.get_portfolio(strategy_name)

        # --- Risk rules: PositionSizingRule, MinSignalStrengthRule, PostFillCooldownRule ---
        if risk_manager is not None:
            for rule in risk_manager._rules:
                if isinstance(rule, PositionSizingRule):
                    if profile.position_size_percent is not None:
                        rule._size_percent = profile.position_size_percent
                        changes[f"{prefix}.position_sizing._size_percent"] = str(
                            profile.position_size_percent
                        )

                elif isinstance(rule, MinSignalStrengthRule):
                    if profile.min_signal_strength is not None:
                        rule._min_strength = profile.min_signal_strength
                        changes[f"{prefix}.min_signal_strength._min_strength"] = str(
                            profile.min_signal_strength
                        )

                elif isinstance(rule, PostFillCooldownRule):
                    if profile.post_fill_cooldown_seconds is not None:
                        rule._cooldown_seconds = profile.post_fill_cooldown_seconds
                        changes[f"{prefix}.post_fill_cooldown._cooldown_seconds"] = (
                            profile.post_fill_cooldown_seconds
                        )

        # --- ExitMonitor ---
        if exit_monitor is not None and isinstance(exit_monitor, ExitMonitor):
            # Warn about SL tightening with open positions before mutating
            if profile.stop_loss_percent is not None:
                old_sl = exit_monitor._stop_loss_pct
                new_sl = profile.stop_loss_percent
                if new_sl < old_sl and portfolio is not None:
                    open_positions = [
                        sym for sym, pos in (portfolio._positions or {}).items()
                        if pos is not None and pos.amount > 0
                    ]
                    if open_positions:
                        self._log.warning(
                            "profile_sl_tightened_with_open_positions",
                            strategy=strategy_name,
                            old_sl=str(old_sl),
                            new_sl=str(new_sl),
                            open_positions=open_positions,
                            note="tighter stop-loss may trigger immediate exit on next tick",
                        )
                exit_monitor._stop_loss_pct = new_sl
                changes[f"{prefix}.exit_monitor._stop_loss_pct"] = str(new_sl)

            if profile.take_profit_percent is not None:
                exit_monitor._take_profit_pct = profile.take_profit_percent
                changes[f"{prefix}.exit_monitor._take_profit_pct"] = str(
                    profile.take_profit_percent
                )

            if profile.max_position_age_minutes is not None:
                # ExitMonitor stores age in seconds internally
                exit_monitor._max_age_seconds = profile.max_position_age_minutes * 60
                changes[f"{prefix}.exit_monitor._max_age_seconds"] = (
                    profile.max_position_age_minutes * 60
                )

            if profile.adaptive_tp is not None:
                exit_monitor._adaptive_tp = profile.adaptive_tp
                changes[f"{prefix}.exit_monitor._adaptive_tp"] = profile.adaptive_tp

            if profile.tp_multiplier is not None:
                exit_monitor._tp_multiplier = profile.tp_multiplier
                changes[f"{prefix}.exit_monitor._tp_multiplier"] = str(
                    profile.tp_multiplier
                )

            if profile.min_tp_percent is not None:
                exit_monitor._min_tp_percent = profile.min_tp_percent
                changes[f"{prefix}.exit_monitor._min_tp_percent"] = str(
                    profile.min_tp_percent
                )

        # --- SignalAggregator ---
        if aggregator is not None and profile.aggregation_threshold is not None:
            aggregator._threshold = profile.aggregation_threshold
            changes[f"{prefix}.aggregator._threshold"] = str(
                profile.aggregation_threshold
            )

        self._log.debug(
            "pipeline_profile_applied",
            strategy=strategy_name,
            changes=changes,
        )
        return changes
