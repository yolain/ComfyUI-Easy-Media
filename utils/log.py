"""Console logging helpers shared by Easy Media nodes."""

from __future__ import annotations


_BOLD_CYAN = "\033[1m\033[36m"
_RESET = "\033[0m"

__all__ = ["log_node_info"]


def log_node_info(node_name: str, message: str | None = None) -> None:
    """Display an informational node message in the ComfyUI console."""
    title = node_name.replace(" (EasyMedia)", "")
    suffix = f":{_RESET} {message}" if message is not None else _RESET
    print(f"{_BOLD_CYAN}[EasyMedia] {title}{suffix}", flush=True)  # noqa: T201
