"""Persistent state helpers for the optional desktop application."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
from urllib.parse import quote, urlencode


_LAYOUT_START = "/* MODVIZ_SAVED_LAYOUT_START */"
_LAYOUT_END = "/* MODVIZ_SAVED_LAYOUT_END */"


def recent_models(settings, limit: int = 8) -> list[Path]:
    """Return existing recent model paths and remove stale entries."""
    value = settings.value("files/recent_models", [])
    values = [value] if isinstance(value, str) else list(value or [])
    paths = [Path(item).expanduser().resolve() for item in values]
    existing = [path for path in paths if path.is_file()]
    if len(existing) != len(paths):
        settings.setValue("files/recent_models", [str(path) for path in existing])
    return existing[:limit]


def remember_recent_model(settings, path: Path, limit: int = 8) -> None:
    """Move *path* to the front of the recent-model list."""
    path = path.expanduser().resolve()
    previous = [item for item in recent_models(settings, limit) if item != path]
    settings.setValue("files/recent_models", [str(path), *map(str, previous[: limit - 1])])


def embed_layout_html(document: str, layout: str) -> str:
    """Embed a live editor layout into a self-contained ModViz HTML file."""
    document = re.sub(
        rf"\s*{re.escape(_LAYOUT_START)}.*?{re.escape(_LAYOUT_END)}\s*",
        "\n",
        document,
        flags=re.DOTALL,
    )
    before, closing, after = document.rpartition("</script>")
    if not closing:
        raise ValueError("This is not a complete ModViz HTML document")
    encoded = json.dumps(layout).replace("<", "\\u003c")
    saved = (
        f"\n{_LAYOUT_START}\n"
        f"setTimeout(()=>importLayoutState({encoded}),0);\n"
        f"{_LAYOUT_END}\n"
    )
    return before + saved + closing + after


def welcome_html(paths: list[Path]) -> str:
    """Build the no-model screen with safe recent-model links."""
    links = []
    for path in paths:
        href = "https://modviz.local/open?" + urlencode({"path": str(path)}, quote_via=quote)
        links.append(
            '<a class="recent" href="{}"><span>{}</span><small>{}</small></a>'.format(
                html.escape(href, quote=True), html.escape(path.name), html.escape(str(path.parent))
            )
        )
    recent = "".join(links) or '<div class="none">No recent models yet</div>'
    return f"""<!doctype html><style>
    body{{margin:0;display:grid;place-items:center;height:100vh;background:#f5f6f8;
    color:#17202a;font-family:system-ui,sans-serif}}.card{{width:min(680px,calc(100vw - 60px));padding:38px;
    background:white;border:1px solid #ccd2da;border-radius:12px;box-shadow:0 8px 30px #23364b18}}
    h1{{font-family:Georgia,serif;margin:0 0 10px}}p{{line-height:1.55;color:#52606f;margin:0 0 24px}}
    h2{{margin:24px 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#687382}}
    .recent{{display:flex;flex-direction:column;padding:12px 14px;margin:7px 0;border:1px solid #d5dce5;
    border-radius:7px;color:#244d76;text-decoration:none;background:#fbfdff}}.recent:hover{{border-color:#537ca5;background:#eef4fa}}
    .recent span{{font-weight:650}}.recent small{{margin-top:3px;color:#75808d;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
    .browse{{display:inline-block;margin-top:15px;padding:9px 14px;border-radius:6px;background:#244d76;color:white;text-decoration:none}}
    .none{{color:#8a929c;font-style:italic;padding:8px 0}}
    </style><div class=card><h1>ModViz Desktop</h1><p>Open a trusted PyTorch model or a saved
    ModViz HTML diagram, then inspect and edit the interactive network flow.</p><h2>Recent files</h2>{recent}
    <a class="browse" href="https://modviz.local/browse">Open another file…</a></div>"""
