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
    candidates = (source.parent.parent, source.parent.parent.parent)
    for root in candidates:
        package = root / "src" / "vessel_gnn" if (root / "src" / "vessel_gnn").is_dir() else root / "vessel_gnn"
        config = root / "configs" / "config.yaml" if (root / "configs" / "config.yaml").is_file() else root / "config.yaml"
        if config.is_file() and (package / "graph_ops.py").is_file() and (package / "losses.py").is_file():
            return root
    return None


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
        python_path = [str(repository), str(project), str(project / "src")]
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
