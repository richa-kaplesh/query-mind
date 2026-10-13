"""Tiny helper for logging process memory (Linux /proc; returns None elsewhere)."""
import os


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB, or None if it can't be read."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB""""Tiny helper for logging process memory (Linux /proc; returns None elsewhere)."""
import os


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB, or None if it can't be read."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB""""Tiny helper for logging process memory (Linux /proc; returns None elsewhere)."""
import os


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB, or None if it can't be read."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB""""Tiny helper for logging process memory (Linux /proc; returns None elsewhere)."""
import os


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB, or None if it can't be read."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB""""Tiny helper for logging process memory (Linux /proc; returns None elsewhere)."""
import os


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB, or None if it can't be read."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB"