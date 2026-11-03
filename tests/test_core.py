import unittest
from pathlib import Path

from torch_flow_viz.model_loader import parse_json_object
from torch_flow_viz.project_adapter import vessel_project_root
from torch_flow_viz.render import render_html
from torch_flow_viz.tracer import parse_shape


class CoreTests(unittest.TestCase):
    def test_desktop_entrypoint_is_packaged(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        desktop = (root / "torch_flow_viz" / "desktop.py").read_text(encoding="utf-8")
        self.assertIn('desktop = ["PySide6>=6.6"]', pyproject)
        self.assertIn('torch-flow-viz-desktop = "torch_flow_viz.desktop:main"', pyproject)
        self.assertIn("class MainWindow", desktop)
        self.assertIn("QWebEngineView", desktop)
        self.assertIn("downloadRequested", desktop)
        self.assertIn("QComboBox,", desktop)
        self.assertIn("self.device = QComboBox()", desktop)
        self.assertIn("completed = Signal()", desktop)
        self.assertIn("QThread.currentThread().quit()", desktop)
        self.assertIn("thread.finished.connect(self._trace_thread_finished)", desktop)
        self.assertIn("render_supported_project", desktop)
        self.assertIn("def _trace_thread_finished", desktop)
        self.assertIn("except RuntimeError", desktop)

    def test_vessel_project_detection(self):
        model = Path("/Users/ifryed/Documents/ChatGPT/NextProj/vessel_gnn/model.py")
        if model.is_file():
            self.assertEqual(vessel_project_root(model), model.parents[1])
        self.assertIsNone(vessel_project_root(Path(__file__)))

    def test_shape(self):
        self.assertEqual(parse_shape("1, 3,224,224"), (1, 3, 224, 224))
        with self.assertRaises(ValueError):
            parse_shape("1,0")

    def test_json_object(self):
        self.assertEqual(parse_json_object('{"classes": 4}', "--x"), {"classes": 4})
        with self.assertRaises(ValueError):
            parse_json_object("[]", "--x")

    def test_html_payload_is_script_safe(self):
        html = render_html({"name": "</script>", "nodes": [], "edges": [], "total_parameters": 0})
        self.assertNotIn('const graph={"name": "</script>', html)
        self.assertIn("PyTorch network flow", html)
        self.assertIn('id="png"', html)
        self.assertIn('id="pdf"', html)
        self.assertIn('id="drawio"', html)
        self.assertIn("makeDrawio", html)
        self.assertIn('id="reset"', html)
        self.assertIn('id="help"', html)
        self.assertIn('id="helpOverlay"', html)
        self.assertIn("How to use this diagram", html)
        self.assertIn("openHelp", html)
        self.assertIn('id="undo"', html)
        self.assertIn('id="toggleLoss"', html)
        self.assertIn('id="compact"', html)
        self.assertIn("compactLayout", html)
        self.assertIn("levelByKey", html)
        self.assertIn("toggleLossVisibility", html)
        self.assertIn("lossHidden=!showLoss", html)
        self.assertIn("visibleNodes=graph.nodes.filter", html)
        self.assertIn('id="alignH"', html)
        self.assertIn('id="alignV"', html)
        self.assertIn('id="group"', html)
        self.assertIn('id="ungroup"', html)
        self.assertIn('id="collapse"', html)
        self.assertIn('id="rename"', html)
        self.assertIn('id="contextTools"', html)
        self.assertIn("positionContextToolbar", html)
        self.assertIn("tools.hidden=!(multi||groups.length)", html)
        self.assertIn("minimizeIcon", html)
        self.assertIn("expandIcon", html)
        self.assertIn('aria-label="Group selected objects"', html)
        self.assertIn('data-tooltip="Align horizontally"', html)
        self.assertIn("collapse.setAttribute('data-tooltip'", html)
        self.assertIn("stroke-width:1.2", html)
        self.assertIn("groupSelected", html)
        self.assertIn("ungroupSelected", html)
        self.assertIn("toggleCollapseSelected", html)
        self.assertIn("renameSelectedGroup", html)
        self.assertIn("Group name cannot be empty", html)
        self.assertIn("boxForNode", html)
        self.assertIn("visibleRoutingBoxes", html)
        self.assertIn("endpointPortOffset", html)
        self.assertIn("sideAnchor", html)
        self.assertIn("chooseCollapsedSide", html)
        self.assertIn("collapsedSideUsage", html)
        self.assertIn("startItemDrag", html)
        self.assertIn("refreshLayout", html)
        self.assertIn("routeEdge", html)
        self.assertIn("horizontalPathClear", html)
        self.assertIn("startMarquee", html)
        self.assertIn("selection-marquee", html)
        self.assertIn("p.x>=left&&p.x+W<=right", html)
        self.assertIn("selectedGroups", html)
        self.assertIn("declaredGroups", html)
        self.assertIn("declaredMemberIds", html)
        self.assertIn("definition.collapsed", html)
        self.assertIn("selectGroup", html)
        self.assertIn("groupRoutingObstacles", html)
        self.assertIn("routeWithGroupGateways", html)
        self.assertIn("reduceRouteBends", html)
        self.assertIn("assignPorts", html)
        self.assertIn("sideVector", html)
        self.assertIn("segmentFollowsSide", html)
        self.assertIn("edgePort", html)
        self.assertIn("boundaryFacingSide", html)
        self.assertIn("groupedEndpointSide", html)
        self.assertIn("preferred==='top'||preferred==='bottom'?preferred", html)
        self.assertIn("nearestGroupSide", html)
        self.assertIn("segmentConflicts", html)
        self.assertIn("routedSegments", html)
        self.assertIn("clearance=10", html)
        self.assertIn("gatewayLeg", html)
        self.assertIn("undoLayout", html)
        self.assertIn("layoutHistory", html)
        self.assertIn("stroke-dasharray:7 4", html)
        self.assertIn("graphX=(mx-tx)/oldScale", html)
        self.assertIn("Math.abs(e.deltaX)>Math.abs(e.deltaY)", html)


if __name__ == "__main__":
    unittest.main()

