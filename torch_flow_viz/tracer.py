from __future__ import annotations

import operator
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class GraphNode:
    id: str
    label: str
    op: str
    target: str
    shape: str
    dtype: str
    parameters: int
    details: str
    category: str


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str


def trace_model(model, inputs: list[Any]) -> dict[str, Any]:
    """Symbolically trace a model and enrich its nodes by executing shape propagation."""
    try:
        import torch
        from torch.fx import GraphModule, Node, Tracer
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install it with: pip install torch") from exc

    model.eval()
    class NetworkTracer(Tracer):
        def is_leaf_module(self, module, qualified_name):
            package = module.__class__.__module__
            name = module.__class__.__name__.lower()
            # PyG and DGL layers often contain Python control flow that cannot be
            # entered by FX. Treat the layer as one meaningful graph operation.
            if package.startswith(("torch_geometric.nn", "dgl.nn")) or any(
                token in name for token in ("graphconv", "gcnconv", "gatconv", "sageconv", "messagepassing")
            ):
                return True
            return super().is_leaf_module(module, qualified_name)

    graph_module = GraphModule(model, NetworkTracer().trace(model))
    with torch.no_grad():
        ShapeProp(graph_module).propagate(*inputs)

    modules = dict(graph_module.named_modules())
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for node in graph_module.graph.nodes:
        shape, dtype = _tensor_metadata(node.meta.get("tensor_meta"))
        target = _target_name(node.target)
        label = _node_label(node, modules)
        params = 0
        details = ""
        if node.op == "call_module" and str(node.target) in modules:
            child = modules[str(node.target)]
            params = sum(p.numel() for p in child.parameters(recurse=False))
            details = repr(child)
        elif node.op in {"call_function", "call_method"}:
            details = target
        nodes.append(
            GraphNode(
                node.name,
                label,
                node.op,
                target,
                shape,
                dtype,
                params,
                details,
                _category(node, modules),
            )
        )

        for parent in _input_nodes(node.args, node.kwargs, Node):
            parent_shape, _ = _tensor_metadata(parent.meta.get("tensor_meta"))
            edges.append(GraphEdge(parent.name, node.name, parent_shape))

    return {
        "name": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
    }


def make_inputs(
    shapes: list[tuple[int, ...]], dtype: str | list[str] = "float32", device: str = "cpu"
) -> list[Any]:
    import torch

    dtypes = [dtype] * len(shapes) if isinstance(dtype, str) else dtype
    if len(dtypes) != len(shapes):
        raise ValueError("The number of input dtypes must match the number of input shapes")
    torch_dtypes = []
    for name in dtypes:
        if not hasattr(torch, name) or not isinstance(getattr(torch, name), torch.dtype):
            raise ValueError(f"Unknown torch dtype: {name}")
        torch_dtypes.append(getattr(torch, name))
    return [torch.zeros(shape, dtype=item_dtype, device=device) for shape, item_dtype in zip(shapes, torch_dtypes)]


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid input shape {value!r}; use comma-separated integers") from exc
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"Invalid input shape {value!r}; dimensions must be positive")
    return shape


def _input_nodes(args, kwargs, node_type):
    found = []

    def visit(value):
        if isinstance(value, node_type):
            found.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(args)
    visit(kwargs)
    # Preserve operand order while avoiding duplicate edges.
    return list(dict.fromkeys(found))


def _target_name(target) -> str:
    if target is operator.getitem:
        return "getitem"
    return getattr(target, "__name__", str(target))


def _node_label(node, modules) -> str:
    if node.op == "call_module" and str(node.target) in modules:
        return f"{node.target}\n{modules[str(node.target)].__class__.__name__}"
    if node.op == "placeholder":
        return f"Input\n{node.target}"
    if node.op == "output":
        return "Output"
    return _target_name(node.target)


def _category(node, modules) -> str:
    if node.op in {"placeholder", "output"}:
        return "io"
    if node.op == "call_module" and str(node.target) in modules:
        module = modules[str(node.target)]
        name = module.__class__.__name__.lower()
        package = module.__class__.__module__.lower()
        if "loss" in name:
            return "loss"
        if "torch_geometric" in package or package.startswith("dgl.") or any(
            token in name for token in ("graphconv", "gcnconv", "gatconv", "sageconv", "messagepassing")
        ):
            return "graph"
        if any(token in name for token in ("conv", "convolution")):
            return "convolution"
        if any(token in name for token in ("linear", "bilinear")):
            return "linear"
        if any(token in name for token in ("relu", "gelu", "sigmoid", "tanh", "softmax", "activation", "silu", "mish")):
            return "activation"
        if "pool" in name:
            return "pooling"
        if any(token in name for token in ("norm", "batchnorm", "layernorm", "groupnorm")):
            return "normalization"
        if any(token in name for token in ("attention", "transformer")):
            return "attention"
        if any(token in name for token in ("lstm", "gru", "rnn")):
            return "recurrent"
        if any(token in name for token in ("dropout", "embedding")):
            return name.rstrip("0123456789")
        return "module"
    target = _target_name(node.target).lower()
    if "loss" in target or any(token in target for token in ("cross_entropy", "kl_div", "nll_loss")):
        return "loss"
    if any(token in target for token in ("reshape", "flatten", "view", "permute", "transpose", "squeeze")):
        return "reshape"
    if any(token in target for token in ("relu", "gelu", "sigmoid", "tanh", "softmax", "silu")):
        return "activation"
    if any(token in target for token in ("add", "sub", "mul", "div", "matmul", "cat")):
        return "arithmetic"
    return "function"


def _tensor_metadata(meta) -> tuple[str, str]:
    if meta is None:
        return "", ""
    if hasattr(meta, "shape"):
        shape = "×".join(str(dim) for dim in meta.shape)
        dtype = str(meta.dtype).replace("torch.", "")
        return shape, dtype
    if isinstance(meta, (tuple, list)):
        values = [_tensor_metadata(item) for item in meta]
        return " | ".join(item[0] for item in values), " | ".join(item[1] for item in values)
    return "", ""

