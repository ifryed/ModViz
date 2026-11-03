# ModViz

Turn a PyTorch model into a **self-contained interactive HTML** network-flow diagram — ops, shapes, dtypes, parameter counts, and stage panels ready for papers or slides.

![Hybrid CNN + GNN overview](docs/images/hybrid-overview.png)

Package name: `torch-flow-viz`. Open any generated HTML and click **Help** for the full in-diagram guide.

## Features

| | |
|---|---|
| **Trace** | `torch.fx` data-flow (not just nested modules) |
| **Arrange** | Drag layers / stages · Shift-click or marquee select · Align · Group / Minimize |
| **Inspect** | Click a layer for shape, dtype, params, and module details |
| **Export** | PNG · SVG · PDF · draw.io (full figure, not just the viewport) |
| **Undo** | Layout history with Ctrl/Cmd+Z |

![Layer details panel](docs/images/hybrid-details.png)

![Stage groups and floating tools](docs/images/hybrid-group.png)

## Install

Python 3.9+ and PyTorch 2.x:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
torch-flow-viz examples/simple_cnn.py \
  --input-shape 1,3,224,224 \
  -o simple_cnn.html
```

Ready-made hybrid demo: open [`examples/hybrid_cnn_gnn_visualization.html`](examples/hybrid_cnn_gnn_visualization.html).

```bash
torch-flow-viz examples/hybrid_cnn_gnn.py --model HybridCNNGNN \
  --input-shape 8,3,32,32 --input-dtype float32 \
  --input-shape 8,8 --input-dtype float32 \
  --input-shape 8 --input-dtype int64 \
  -o examples/hybrid_cnn_gnn_visualization.html
```

## Your own model

Expose `model = MyNetwork()` in a `.py` file, or point at a class:

```bash
torch-flow-viz my_network.py --model ResNet \
  --init-args '{"num_classes": 5}' \
  --input-shape 1,3,224,224 -o resnet.html
```

Repeat `--input-shape` / `--input-dtype` for multiple inputs. From Python:

```python
from torch_flow_viz import visualize
visualize(model, [torch.zeros(1, 3, 224, 224)], "network.html")
```

Losses appear only if they are part of `forward`. Loading a model file **executes** its Python — use trusted local sources. Models with heavy data-dependent control flow may need an FX leaf wrap.
