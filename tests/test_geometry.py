"""Behavioral regressions for parallel connectors, compact, reset and groups."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from torch_flow_viz.render import render_html


def repeated_graph():
    nodes, edges, groups = [], [], []

    def node(identifier, category="graph", op="call_module"):
        nodes.append(dict(id=identifier, label=identifier, target=identifier,
                          shape="16×64", dtype="float32", parameters=0,
                          details="", category=category, op=op))

    def edge(source, target):
        edges.append(dict(source=source, target=target, label="16×64"))

    for identifier in ("features", "edge_index", "targets"):
        node(identifier, "io", "placeholder")
    node("projection", "linear")
    edge("features", "projection")
    embedding, topology, stats = "projection", "edge_index", "features"
    assignments, edge_heads, node_heads = [], [], []
    for i in range(4):
        names = [f"s{i}_{part}" for part in ("gat", "norm", "unique", "edge_head",
                                           "match", "mean", "stats", "fusion",
                                           "topology", "node_head")]
        gat, norm, unique, head, match, mean, pooled_stats, fusion, pool, pred = names
        for name in names:
            node(name)
        groups.append(dict(name=f"Stage {i + 1}", members=names, collapsed=True))
        for source, target in (
            (embedding, gat), (topology, gat), (gat, norm), (topology, unique),
            (norm, unique), (norm, head), (unique, head), (head, match),
            (unique, match), (norm, mean), (match, mean), (stats, pooled_stats),
            (match, pooled_stats), (mean, fusion), (pooled_stats, fusion),
            (topology, pool), (match, pool), (fusion, pred),
        ):
            edge(source, target)
        embedding, topology, stats = fusion, pool, pooled_stats
        assignments.append(match)
        edge_heads.append(head)
        node_heads.append(pred)
    node("unpool", "reshape")
    for name in ("edge_loss", "node_loss", "pixel_loss", "total_loss"):
        node(name, "loss")
    node("output", "io", "output")
    for source in assignments + [node_heads[-1]]:
        edge(source, "unpool")
    for source in edge_heads:
        edge(source, "edge_loss")
    for source in node_heads:
        edge(source, "node_loss")
    for source, target in (
        ("unpool", "pixel_loss"), ("targets", "pixel_loss"),
        ("edge_loss", "total_loss"), ("node_loss", "total_loss"),
        ("pixel_loss", "total_loss"), ("unpool", "output"), ("total_loss", "output"),
    ):
        edge(source, target)
    return dict(name="Repeated graph regression", nodes=nodes, edges=edges,
                groups=groups, total_parameters=0)


class GeometryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for renderer tests")
    def test_repeated_groups_route_without_merges_or_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "diagram.html"
            path.write_text(render_html(repeated_graph()), encoding="utf-8")
            result = subprocess.run(
                ["node", str(Path(__file__).with_name("check_geometry.cjs")), str(path)],
                text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
