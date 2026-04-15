"""Task 18: 60s stream gap during RTH → new entries denied until resume."""
import pytest


@pytest.mark.skip(reason="needs stream-stale detection infra (not yet built) — deferred")
def test_new_entries_denied_during_stream_stale():
    pass
