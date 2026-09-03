"""The terminal adapter's user-facing strings."""

from __future__ import annotations

from kasa import __version__
from kasa.adapters.cli.repl import banner


def test_the_banner_names_the_running_version() -> None:
    """#48: a hand-maintained label told users of a memory build there was none.

    The point is not the words but where they come from — a version written
    into a string is one that goes stale without anything failing.
    """
    assert __version__ in banner()
    assert "v0" not in banner()
    assert "No memory yet" not in banner()
