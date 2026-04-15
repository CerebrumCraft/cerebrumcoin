"""Manual live ORB smoke test — not in CI.

Run this during an RTH window (ideally right at the 09:30 open) to see
the full ORB pipeline respond to real Alpaca data. Requires --live-alpaca.

@decision DEC-STOCKS-004
"""

import pytest

pytestmark = pytest.mark.live_alpaca


@pytest.mark.skip(
    reason=(
        "manual-only — not in CI. Run explicitly with "
        "pytest tests/live/test_live_orb_smoke.py --live-alpaca -s"
    )
)
def test_live_orb_smoke_placeholder() -> None:
    """Placeholder. See docstring for manual execution instructions."""
    pass
