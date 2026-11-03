"""Generate the ready-to-open hybrid CNN + GNN visualization without PyTorch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_flow_viz.render import render_html


def node(id, label, op, target, shape, category, parameters=0, details="", dtype="float32"):
    return {
        "id": id,
        "label": label,
        "op": op,
        "target": target,
        "shape": shape,
        "dtype": dtype,
        "parameters": parameters,
        "details": details,
        "category": category,
    }


nodes = [
    node("images", "Input\nnode_images", "placeholder", "node_images", "8×3×32×32", "io"),
    node("adj", "Input\nadjacency", "placeholder", "adjacency", "8×8", "io"),
    node("target", "Input\ntargets", "placeholder", "targets", "8", "io", dtype="int64"),
    node("conv1", "conv1\nConv2d", "call_module", "conv1", "8×16×32×32", "convolution", 448, "Conv2d(3, 16, kernel_size=3, padding=1)"),
    node("relu1", "relu1\nReLU", "call_module", "relu1", "8×16×32×32", "activation", details="ReLU()"),
    node("pool", "pool\nMaxPool2d", "call_module", "pool", "8×16×16×16", "pooling", details="MaxPool2d(2)"),
    node("conv2", "conv2\nConv2d", "call_module", "conv2", "8×32×16×16", "convolution", 4640, "Conv2d(16, 32, kernel_size=3, padding=1)"),
    node("relu2", "relu2\nReLU", "call_module", "relu2", "8×32×16×16", "activation", details="ReLU()"),
    node("avg", "global_pool\nAdaptiveAvgPool2d", "call_module", "global_pool", "8×32×1×1", "pooling", details="AdaptiveAvgPool2d(1)"),
    node("flat", "flatten", "call_function", "flatten", "8×32", "reshape"),
    node("graph1", "graph1\nDenseGraphConv", "call_module", "graph1", "8×24", "graph", 792, "DenseGraphConv(32, 24)"),
    node("grelu", "graph_relu\nReLU", "call_module", "graph_relu", "8×24", "activation", details="ReLU()"),
    node("graph2", "graph2\nDenseGraphConv", "call_module", "graph2", "8×4", "graph", 100, "DenseGraphConv(24, 4)"),
    node("loss", "loss_fn\nCrossEntropyLoss", "call_module", "loss_fn", "scalar", "loss", details="CrossEntropyLoss()"),
    node("output", "Outputs\nprediction + loss", "output", "output", "8×4 | scalar", "io"),
]

links = [
    ("images", "conv1"), ("conv1", "relu1"), ("relu1", "pool"),
    ("pool", "conv2"), ("conv2", "relu2"), ("relu2", "avg"),
    ("avg", "flat"), ("flat", "graph1"), ("adj", "graph1"),
    ("graph1", "grelu"), ("grelu", "graph2"), ("adj", "graph2"),
    ("graph2", "loss"), ("target", "loss"),
    ("graph2", "output"), ("loss", "output"),
]

graph = {
    "name": "HybridCNNGNN",
    "total_parameters": 5980,
    "trainable_parameters": 5980,
    "nodes": nodes,
    "edges": [
        {"source": source, "target": target, "label": next(n["shape"] for n in nodes if n["id"] == source)}
        for source, target in links
    ],
}
destination = Path(__file__).with_name("hybrid_cnn_gnn_visualization.html")
destination.write_text(render_html(graph), encoding="utf-8")
print(f"Created {destination}")

