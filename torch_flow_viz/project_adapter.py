"""Adapters for projects whose inputs or control flow cannot be expressed by FX."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def vessel_project_root(model_path: str | Path) -> Path | None:
    """Return the project root when *model_path* is the supported Vessel GNN."""
    source = Path(model_path).expanduser().resolve()
    if source.name != "model.py" or source.parent.name != "vessel_gnn":
        return None
    root = source.parent.parent
    required = (
        root / "config.yaml",
        root / "vessel_gnn" / "graph_ops.py",
        root / "vessel_gnn" / "losses.py",
    )
    return root if all(path.is_file() for path in required) else None


def render_supported_project(model_path: str | Path) -> tuple[str, str] | None:
    """Render a known data-dependent project, or return ``None``."""
    project = vessel_project_root(model_path)
    if project is None:
        return None

    repository = Path(__file__).resolve().parents[1]
    generator = repository / "examples" / "create_nextproj_vessel_demo.py"
    if not generator.is_file():
        raise RuntimeError(
            "The Vessel GNN adapter is available, but its generator is missing: "
            f"{generator}"
        )

    with tempfile.TemporaryDirectory(prefix="modviz-vessel-") as directory:
        output = Path(directory) / "vessel_gnn.html"
        environment = os.environ.copy()
        python_path = [str(repository), str(project)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        process = subprocess.run(
            [sys.executable, str(generator), str(project), "-o", str(output)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if process.returncode:
            details = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"Vessel GNN adapter failed:\n{details}")
        return output.read_text(encoding="utf-8"), "HierarchicalVesselGNN"
