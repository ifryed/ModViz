"""Create a diagram for the Hierarchical Vessel GNN project.

The model contains intentional data-dependent hard graph matching, which cannot
be represented by torch.fx. This generator executes the real model on a small
pixel graph, then describes the four repeated stages and the losses present in
the project's source files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_flow_viz.render import render_html


def node(identifier, label, target, shape, category, parameters=0, details="", op="call_module", dtype="float32"):
    return {
        "id": identifier,
        "label": label,
        "op": op,
        "target": target,
        "shape": shape,
        "dtype": dtype,
        "parameters": parameters,
        "details": details,
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("nextproj_vessel_gnn.html"))
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    sys.path.insert(0, str(project))

    import torch
    from torch_geometric.data import Data
    from vessel_gnn.model import HierarchicalVesselGNN

    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    loss_source = (project / "vessel_gnn" / "losses.py").read_text(encoding="utf-8")
    required_losses = ("edge_loss", "node_loss", "pixel_loss", "binary_cross_entropy_with_logits")
    if not all(name in loss_source for name in required_losses):
        raise RuntimeError("The expected hierarchical losses were not found in vessel_gnn/losses.py")

    model = HierarchicalVesselGNN(config["model"]).eval()
    width = height = 4
    count = width * height
    coordinates = torch.tensor([(x, y) for y in range(height) for x in range(width)], dtype=torch.float32)
    edges = [
        (source, target)
        for source in range(count)
        for target in range(count)
        if source != target
        and abs(source % width - target % width) <= 1
        and abs(source // width - target // width) <= 1
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ones = torch.ones(count)
    data = Data(
        edge_index=edge_index,
        count=ones,
        positive_count=torch.zeros(count),
        rgb_sum=torch.rand(count, 3),
        x_sum=coordinates[:, 0],
        y_sum=coordinates[:, 1],
        xx_sum=coordinates[:, 0].square(),
        yy_sum=coordinates[:, 1].square(),
        xy_sum=coordinates[:, 0] * coordinates[:, 1],
        pixel_y=torch.zeros(count),
        batch=torch.zeros(count, dtype=torch.long),
    )
    with torch.no_grad():
        result = model(data)

    nodes = [
        node("stats", "Input\nregion statistics", "data.statistics", f"{count}×8", "io", op="placeholder"),
        node("edges", "Input\nedge_index", "data.edge_index", f"2×{edge_index.shape[1]}", "io", op="placeholder", dtype="int64"),
        node("targets", "Input\npixel_y", "data.pixel_y", str(count), "io", op="placeholder"),
        node(
            "projection",
            "input_projection\nLinear + ReLU + LayerNorm",
            "input_projection",
            f"{count}×{config['model']['hidden_dim']}",
            "linear",
            sum(parameter.numel() for parameter in model.input_projection.parameters()),
            repr(model.input_projection),
        ),
    ]
    links = [("stats", "projection")]
    embedding_source = "projection"
    topology_source = "edges"
    stats_source = "stats"
    assignment_sources = []
    edge_heads = []
    node_heads = []

    hidden = int(config["model"]["hidden_dim"])
    for stage_index, stage in enumerate(model.stages, start=1):
        prefix = f"stage{stage_index}"
        gat = f"{prefix}_gat"
        norm = f"{prefix}_norm"
        unique = f"{prefix}_unique"
        edge_head = f"{prefix}_edge_head"
        matching = f"{prefix}_matching"
        mean_pool = f"{prefix}_mean_pool"
        stats_pool = f"{prefix}_stats_pool"
        fusion = f"{prefix}_fusion"
        topology = f"{prefix}_topology"
        node_head = f"{prefix}_node_head"
        stage_nodes = int(result["node_counts"][stage_index])
        nodes.extend(
            [
                node(gat, f"Stage {stage_index} GAT\nGATv2Conv", f"stages.{stage_index-1}.gat", f"{stage_nodes}×{hidden}", "graph", sum(p.numel() for p in stage.gat.parameters()), repr(stage.gat)),
                node(norm, "residual + norm\nLayerNorm", f"stages.{stage_index-1}.norm", f"{stage_nodes}×{hidden}", "normalization", sum(p.numel() for p in stage.norm.parameters()), repr(stage.norm)),
                node(unique, "canonical undirected edges", "undirected_edges", f"2×E{stage_index}", "graph", op="call_function"),
                node(edge_head, "symmetric edge scorer\nSymmetricEdgeMLP", f"stages.{stage_index-1}.edge_mlp", f"E{stage_index}", "graph", sum(p.numel() for p in stage.edge_mlp.parameters()), repr(stage.edge_mlp)),
                node(matching, f"hard matching\nthreshold {config['model']['merge_thresholds'][stage_index-1]:.2f}", "greedy_one_neighbor_matching", f"{stage_nodes}", "graph", op="call_function", dtype="int64"),
                node(mean_pool, "mean embedding pool\nscatter(mean)", "scatter", f"{stage_nodes}×{hidden}", "pooling", op="call_function"),
                node(stats_pool, "additive region stats\nRegionStats.pool", "RegionStats.pool", f"{stage_nodes}×9", "pooling", op="call_function"),
                node(fusion, "pool fusion\nLinear + ReLU + Norm", f"stages.{stage_index-1}.pool_fusion", f"{stage_nodes}×{hidden}", "linear", sum(p.numel() for p in stage.pool_fusion.parameters()), repr(stage.pool_fusion)),
                node(topology, "topology-preserving pool", "pooled_topology", f"2×E{stage_index+1}", "graph", op="call_function", dtype="int64"),
                node(node_head, "node prediction\nLinear", f"stages.{stage_index-1}.node_head", f"{stage_nodes}", "linear", sum(p.numel() for p in stage.node_head.parameters()), repr(stage.node_head)),
            ]
        )
        links.extend(
            [
                (embedding_source, gat), (topology_source, gat), (gat, norm),
                (topology_source, unique), (norm, unique), (norm, edge_head), (unique, edge_head),
                (edge_head, matching), (unique, matching), (norm, mean_pool), (matching, mean_pool),
                (stats_source, stats_pool), (matching, stats_pool), (mean_pool, fusion), (stats_pool, fusion),
                (topology_source, topology), (matching, topology), (fusion, node_head),
            ]
        )
        embedding_source, topology_source, stats_source = fusion, topology, stats_pool
        assignment_sources.append(matching)
        edge_heads.append(edge_head)
        node_heads.append(node_head)

    nodes.extend(
        [
            node("unpool", "pixel unpooling", "pixel_to_node", str(count), "reshape", op="call_function"),
            node("edge_loss", "edge_loss\nBCEWithLogits", "hierarchical_loss.edge_loss", "scalar", "loss", details="Mean binary cross-entropy across all stages", op="call_function"),
            node("node_loss", "node_loss\nAuxiliary weighted BCE", "hierarchical_loss.node_loss", "scalar", "loss", details=f"Auxiliary weights: {config['loss']['aux_stage_weights']}", op="call_function"),
            node("pixel_loss", "pixel_loss\nBCEWithLogits", "hierarchical_loss.pixel_loss", "scalar", "loss", details=f"pos_weight={config['loss']['pixel_pos_weight']}", op="call_function"),
            node("total_loss", "total loss\nweighted sum", "hierarchical_loss.total", "scalar", "loss", details=f"edge={config['loss']['edge_weight']}, node={config['loss']['node_weight']}, pixel={config['loss']['pixel_weight']}", op="call_function"),
            node("output", "Outputs\npixel logits + loss", "output", f"{count} | scalar", "io", op="output"),
        ]
    )
    links.append((node_heads[-1], "unpool"))
    links.extend((assignment, "unpool") for assignment in assignment_sources)
    links.extend((head, "edge_loss") for head in edge_heads)
    links.extend((head, "node_loss") for head in node_heads)
    links.extend([("unpool", "pixel_loss"), ("targets", "pixel_loss"), ("edge_loss", "total_loss"), ("node_loss", "total_loss"), ("pixel_loss", "total_loss"), ("unpool", "output"), ("total_loss", "output")])

    by_id = {item["id"]: item for item in nodes}
    repeated_stage_groups = [
        {
            "name": f"Coarsening Stage {stage_index}",
            "members": [item["id"] for item in nodes if item["id"].startswith(f"stage{stage_index}_")],
            "style": ["#fbf4f8", "#b984a1"],
            "collapsed": True,
        }
        for stage_index in range(1, int(config["model"]["num_stages"]) + 1)
    ]
    graph = {
        "name": "HierarchicalVesselGNN",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "nodes": nodes,
        "edges": [{"source": source, "target": target, "label": by_id[source]["shape"]} for source, target in links],
        "groups": repeated_stage_groups,
    }
    destination = args.output.expanduser().resolve()
    destination.write_text(render_html(graph), encoding="utf-8")
    print(f"Created {destination}")
    print(f"Verified runtime node counts: {result['node_counts']}")


if __name__ == "__main__":
    main()
