from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def parse_json_object(value: str, option: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option} must be a JSON object")
    return parsed


def load_model(path: str | Path, symbol: str | None = None, init_args: dict[str, Any] | None = None):
    """Load an nn.Module instance, class, or factory from a Python source file."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install it with: pip install torch") from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source}")

    module_name = f"_torch_flow_model_{source.stem}_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model file: {source}")
    module = importlib.util.module_from_spec(spec)
    # Make sibling imports work just as they do when the model file is run directly.
    original_path = list(sys.path)
    sys.modules[module_name] = module
    try:
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    kwargs = init_args or {}

    if symbol:
        if not hasattr(module, symbol):
            raise ValueError(f"{source.name} has no symbol named {symbol!r}")
        candidate = getattr(module, symbol)
        return _instantiate(candidate, kwargs, symbol, nn.Module)

    candidate = getattr(module, "model", None)
    if isinstance(candidate, nn.Module):
        if kwargs:
            raise ValueError("--init-args cannot be used with an already-created `model` object")
        return candidate

    factories = [getattr(module, name, None) for name in ("create_model", "get_model", "build_model")]
    factories = [item for item in factories if callable(item)]
    if len(factories) == 1:
        return _instantiate(factories[0], kwargs, factories[0].__name__, nn.Module)

    classes = [
        value
        for _, value in inspect.getmembers(module, inspect.isclass)
        if issubclass(value, nn.Module) and value is not nn.Module and value.__module__ == module.__name__
    ]
    if len(classes) == 1:
        return _instantiate(classes[0], kwargs, classes[0].__name__, nn.Module)

    choices = [name for name, value in vars(module).items() if callable(value) and not name.startswith("_")]
    hint = f" Candidates: {', '.join(choices)}." if choices else ""
    raise ValueError(
        "Could not choose a model automatically. Expose `model`, define one nn.Module class, "
        "or pass --model SYMBOL." + hint
    )


def _instantiate(candidate, kwargs: dict[str, Any], name: str, module_type):
    if isinstance(candidate, module_type):
        if kwargs:
            raise ValueError(f"--init-args cannot be used because {name!r} is already an instance")
        return candidate
    if not callable(candidate):
        raise TypeError(f"{name!r} is neither an nn.Module nor a callable")
    model = candidate(**kwargs)
    if not isinstance(model, module_type):
        raise TypeError(f"{name!r} returned {type(model).__name__}, not torch.nn.Module")
    return model

