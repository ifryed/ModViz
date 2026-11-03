# ModViz (torch-flow-viz)

Turn a PyTorch model file into a self-contained, interactive HTML data-flow diagram. The graph shows operations, layers, skip connections, tensor shapes, dtypes, and parameter counts. Layer families use distinct colors and shapes: convolution, linear, activation, pooling, normalization, attention, recurrent, graph/message-passing, reshape, and I/O.

Open any generated HTML and click **Help** in the toolbar for an in-diagram guide to navigation, selection, grouping, undo, and export.

The renderer uses a publication-style visual language: serif typography, black
data-flow arrows, restrained pastel layer colors, labeled architecture stages,
and a clean white background suitable for an academic paper.

### Rearranging the figure

Drag any layer to move it independently. Drag the empty area or title band of a
stage panel (for example, **Feature Encoding** or **Graph Reasoning**) to move
the entire group. Orthogonal connectors use obstacle-aware routing and reroute
around layer boxes while dragging.

Shift-click layers to add or remove them from a multi-selection. Drag any
selected layer to move the selection together. Applicable editing controls
float beside the selection instead of occupying the main toolbar. **Align H**
places selected layers on a common horizontal centerline; **Align V** places
them on a common vertical centerline. Exports use the current arrangement. Use **Reset** to
restore the automatic layout and **Fit** to center the complete figure.

Shift-drag on empty canvas to draw a selection marquee. A layer is added only
when its entire box is inside the marquee. A stage panel is selected as one
group when its complete panel is inside the marquee. Stage panels can also be
clicked, Shift-clicked, dragged, and aligned like individual layers.
Use **Group** to turn two or more selected layers/groups into a new group, and
**Ungroup** to dissolve any selected groups without removing their layers.
When one group is selected, use the pencil icon to edit its name. Empty names
and names already used by another group are rejected, and renaming can be
reversed with Undo.
Select a group and choose **Minimize** to replace its contents with one compact
cell. Internal layers and connections are hidden, while external connections
reattach to the compact cell. The cell can be selected, dragged, aligned, and
expanded again with **Expand**. The floating controls only show commands that
apply to the current selection.
Connections to a minimized group are balanced across its left, right, top, and
bottom edges according to the location of the other endpoint and existing port
traffic. This includes top-entry and top-exit ports with perpendicular terminal
segments.
Connectors may use the left, right, top, or bottom edge of a layer. The router
chooses the pair facing the relative direction of the connected objects and
keeps separately spaced ports when several edges share a layer.
Every arrow tail and arrowhead is normal to its object boundary: left/right
ports always use a horizontal terminal segment, while top/bottom ports always
use a vertical terminal segment.

For grouped layers, vertical relationships retain top/bottom ports even when
the layer is not close to that side of its group. The group gateway crosses the
panel boundary separately, so an arrow from an object above can enter the top
of the destination layer instead of being redirected to its bottom. Horizontal
relationships still prefer the nearest practical group boundary. This keeps
connections straight whenever possible and adds corners only for an actual
layer, group, or independent-edge routing constraint.
Selected layers and groups both use a dashed blue boundary. Stage panels are
routing obstacles. External connectors travel around a panel to a dedicated
gateway beside their endpoint, cross the boundary there, and then attach to the
fixed layer port. Only connections whose two endpoints are in the same group
route freely inside that panel.

After obstacle routing, a simplification pass removes unnecessary corners. It
prefers a straight segment, then a single-corner path, whenever that shortcut
does not cross a layer or group and preserves the fixed port directions.

Edges may cross at right angles, but they never share a collinear segment.
Blocks with multiple inputs or outputs receive stable, separately spaced port
slots, so traces do not visually merge and split and remain easy to follow.
Parallel routes maintain at least a small visual corridor between them, and
group gateways receive independent offset lanes rather than reusing one leg.

Layout changes from dragging layers or groups, alignment, and Reset are stored
in a 50-step history. Use **Undo**, Ctrl+Z, or Cmd+Z to restore the previous
layout.

The mouse wheel zooms toward the pointer, keeping the location under the cursor
stationary. A horizontal wheel or trackpad gesture pans sideways; Shift + wheel
provides the same horizontal pan on mice without a dedicated side wheel.

Inputs are drawn according to their meaning and rank: image/tensor inputs use
stacked planes, 2D tensors use a matrix grid, 1D tensors use a vector strip,
scalars use a scalar circle, and adjacency or edge-index inputs use a miniature
node-edge graph. Names such as `adjacency`, `edge_index`, and `graph` take
precedence over rank-based detection.

## Export

Every generated HTML diagram has export buttons in its toolbar:

- **PNG** — high-resolution raster image at up to 3× output scale.
- **SVG** — resolution-independent vector image, ideal for LaTeX or Illustrator.
- **PDF** — downloadable landscape PDF with the complete diagram.
- **draw.io** — editable diagrams.net XML with individual nodes and edges.

Exports contain the full graph rather than only the currently visible viewport.

## Training losses

Losses are shown only when they are part of the supplied model's `forward`
graph. Define the loss module or functional call in the model file and include
its target among the example inputs:

```python
class TrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(32, 4)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, targets):
        logits = self.classifier(features)
        loss = self.loss_fn(logits, targets)
        return logits, loss
```

Loss modules and functional calls are detected and styled automatically. If no
loss exists in the traced file, the visualizer does not add one.

## Install

Python 3.9+ and PyTorch 2.x are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Try the included example

```bash
torch-flow-viz examples/simple_cnn.py \
  --input-shape 1,3,224,224 \
  --output simple_cnn.html
```

Open `simple_cnn.html` in a browser. It is a standalone file and can be shared as-is.

## Hybrid CNN + GNN demo

The included `HybridCNNGNN` processes an image at each graph node with a CNN,
then passes the extracted features through two graph-convolution layers. A
ready-to-open visualization is included at
`examples/hybrid_cnn_gnn_visualization.html`.

Regenerate it from the real PyTorch trace after installing the project:

```bash
torch-flow-viz examples/hybrid_cnn_gnn.py --model HybridCNNGNN \
  --input-shape 8,3,32,32 --input-dtype float32 \
  --input-shape 8,8 --input-dtype float32 \
  --input-shape 8 --input-dtype int64 \
  -o examples/hybrid_cnn_gnn_visualization.html
```

## Use your own model

The model file can expose an already-created instance:

```python
model = MyNetwork()
```

Or it can contain one `torch.nn.Module` class. If the file has several possible models, select a class or factory explicitly:

```bash
torch-flow-viz my_network.py --model ResNet \
  --init-args '{"num_classes": 5}' \
  --input-shape 1,3,224,224 -o resnet.html
```

Repeat `--input-shape` for a model with several tensor inputs:

```bash
torch-flow-viz pair_model.py --model PairNet \
  --input-shape 1,128 --input-shape 1,128
```

GNNs commonly combine floating-point node features with integer edge indices. Supply one dtype for each input:

```bash
torch-flow-viz gnn.py --model GNN \
  --input-shape 100,16 --input-dtype float32 \
  --input-shape 2,500 --input-dtype int64 \
  -o gnn.html
```

CNNs, MLPs, recurrent networks, attention models, and FX-traceable custom models work directly. Layers from PyTorch Geometric and DGL are treated as graph-operation nodes so their internal Python control flow does not obscure the network-level diagram. Install the framework used by the model separately.

You can also call it from Python:

```python
import torch
from torch_flow_viz import visualize
from my_network import MyNetwork

model = MyNetwork()
visualize(model, [torch.zeros(1, 3, 224, 224)], "network.html")
```

## Tracing notes

The program uses `torch.fx`, so it visualizes tensor operations and connections rather than merely listing nested modules. The example inputs are run through FX shape propagation; use safe, local model files you trust, because loading the file executes its Python code.

Some models with data-dependent Python control flow cannot be symbolically traced. In those cases, refactor the conditional into tensor operations or wrap the problematic operation as an FX leaf function.
