"""
CerebrumCoin web dashboard package.

Optional — requires the 'dashboard' extra:
    pip install cerebrumcoin[dashboard]

Provides:
    WebDashboard: FastAPI + WebSocket dashboard for multi-strategy visualization.

Import guard: importing this package without FastAPI installed raises ImportError
with a clear installation hint rather than a cryptic attribute error.
"""

try:
    from cerebrum.dashboard.web import WebDashboard

    __all__ = ["WebDashboard"]
except ImportError as _exc:
    raise ImportError(
        "cerebrum.dashboard requires FastAPI: pip install cerebrumcoin[dashboard]"
    ) from _exc
