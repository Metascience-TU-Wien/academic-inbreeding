# Copyright © 2026 TU Wien

"""Lightweight runtime diagnostics for name disambiguation."""

from __future__ import annotations

from pathlib import Path


def current_rss_megabytes() -> float | None:
    """Read current resident memory on Linux without adding a dependency.

    :return: Resident set size in MiB, or ``None`` when procfs is unavailable.
    """
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None
