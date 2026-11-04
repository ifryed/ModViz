"""Capture the current hybrid demo states used by the README.

Run from the repository root with the desktop extra installed:

    python tests/capture_readme_images.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "hybrid_cnn_gnn_visualization.html"
OUTPUT = ROOT / "docs" / "images"


STATES = [
    ("hybrid-overview.png", "document.querySelector('#fit').click()"),
    (
        "hybrid-details.png",
        "document.querySelector('#fit').click();"
        "document.querySelector('.node').dispatchEvent(new MouseEvent('click',{bubbles:true}))",
    ),
    (
        "hybrid-group.png",
        "document.querySelector('#fit').click();"
        "selectGroup(stageRecords.find(group=>group.name==='Graph Reasoning'),false)",
    ),
    (
        "hybrid-help.png",
        "openHelp();"
        "document.querySelector('.help-dialog').scrollTop="
        "document.querySelector('.help-dialog').scrollHeight",
    ),
]


app = QApplication(["modviz-readme-capture"])
view = QWebEngineView()
view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
view.setFixedSize(1920, 1080)
view.show()
index = 0


def capture_next() -> None:
    global index
    if index == len(STATES):
        app.quit()
        return
    filename, script = STATES[index]

    def save(_result=None) -> None:
        global index
        image = view.grab()
        target = OUTPUT / filename
        if not image.save(str(target), "PNG"):
            raise RuntimeError(f"Could not save {target}")
        print(f"Captured {target.relative_to(ROOT)} ({image.width()}x{image.height()})")
        index += 1
        QTimer.singleShot(150, capture_next)

    view.page().runJavaScript(script, lambda _result: QTimer.singleShot(250, save))


def loaded(ok: bool) -> None:
    if not ok:
        raise RuntimeError(f"Could not load {DEMO}")
    QTimer.singleShot(500, capture_next)


view.loadFinished.connect(loaded)
view.load(QUrl.fromLocalFile(str(DEMO)))
raise SystemExit(app.exec())
