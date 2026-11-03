from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .render import render_html
from .tracer import trace_model


def visualize(model, inputs: Iterable[Any], output: str | Path = "network_flow.html") -> Path:
    """Trace ``model`` with example ``inputs`` and write an interactive HTML diagram."""
    graph = trace_model(model, list(inputs))
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(graph), encoding="utf-8")
    return destination

