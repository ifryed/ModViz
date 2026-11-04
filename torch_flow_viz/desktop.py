"""Standalone ModViz desktop application.

Run with ``torch-flow-viz-desktop`` or ``python -m torch_flow_viz.desktop``.
The Qt WebEngine view hosts the same self-contained editor produced by the
HTML renderer, so desktop and browser diagrams have identical behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback

try:
    from PySide6.QtCore import QObject, QSettings, QThread, QTimer, QUrl, QUrlQuery, Signal, Slot
    from PySide6.QtGui import QAction, QIcon, QKeySequence
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QToolBar,
        QVBoxLayout,
    )
    from PySide6.QtWebEngineCore import QWebEngineDownloadRequest, QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError as exc:  # pragma: no cover - exercised only without desktop extras
    raise SystemExit(
        "The desktop app requires PySide6. Install it with:\n"
        "  pip install -e '.[desktop]'\n"
        "Then run: torch-flow-viz-desktop"
    ) from exc

from .model_loader import load_model, parse_json_object
from .desktop_state import embed_layout_html, recent_models, remember_recent_model, welcome_html
from .project_adapter import render_supported_project, vessel_project_root
from .render import render_html
from .tracer import make_inputs, parse_shape, trace_model


def application_icon() -> QIcon:
    """Return the packaged ModViz application icon."""
    return QIcon(str(Path(__file__).with_name("assets") / "modviz.svg"))


@dataclass(frozen=True)
class TraceRequest:
    path: Path
    symbol: str | None
    init_args: dict
    shapes: list[tuple[int, ...]]
    dtypes: list[str]
    device: str


def _nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


class ModelSetupDialog(QDialog):
    """Collect the same model and example-input options as the command line."""

    def __init__(self, path: Path, settings: QSettings, parent=None) -> None:
        super().__init__(parent)
        self.path = path
        self.settings = settings
        self.request: TraceRequest | None = None
        self.setWindowTitle("Open PyTorch model")
        self.resize(620, 470)

        self.symbol = QLineEdit(str(settings.value("trace/model_symbol", "")))
        self.symbol.setPlaceholderText("Optional class or factory, e.g. MyNetwork")
        self.init_args = QPlainTextEdit(str(settings.value("trace/init_args", "{}")))
        self.shapes = QPlainTextEdit(str(settings.value("trace/input_shapes", "1,3,224,224")))
        self.dtypes = QPlainTextEdit(str(settings.value("trace/input_dtypes", "float32")))
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["cpu", "mps", "cuda"])
        self.device.setCurrentText(str(settings.value("trace/device", "cpu")))

        for editor in (self.init_args, self.shapes, self.dtypes):
            editor.setMaximumHeight(90)

        form = QFormLayout()
        file_value = QLineEdit(str(path))
        file_value.setReadOnly(True)
        form.addRow("Model file", file_value)
        form.addRow("Model symbol", self.symbol)
        form.addRow("Constructor arguments (JSON)", self.init_args)
        form.addRow("Input shapes (one per line)", self.shapes)
        form.addRow("Input dtypes (one per line)", self.dtypes)
        form.addRow("Device", self.device)

        note = QLabel(
            "Each shape line is comma-separated, for example 1,3,224,224. "
            "Use int64 for edge-index or label inputs. The model file is executed, "
            "so open only trusted Python files."
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch(1)
        layout.addWidget(buttons)

    @Slot()
    def _validate(self) -> None:
        try:
            shapes = [parse_shape(line) for line in _nonempty_lines(self.shapes.toPlainText())]
            if not shapes:
                raise ValueError("At least one input shape is required")
            dtypes = _nonempty_lines(self.dtypes.toPlainText())
            if len(dtypes) == 1 and len(shapes) > 1:
                dtypes *= len(shapes)
            if len(dtypes) != len(shapes):
                raise ValueError("Provide one dtype per shape, or one dtype for every input")
            init_args_text = self.init_args.toPlainText().strip() or "{}"
            request = TraceRequest(
                path=self.path,
                symbol=self.symbol.text().strip() or None,
                init_args=parse_json_object(init_args_text, "constructor arguments"),
                shapes=shapes,
                dtypes=dtypes,
                device=self.device.currentText().strip() or "cpu",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Invalid model setup", str(exc))
            return

        self.settings.setValue("trace/model_symbol", self.symbol.text().strip())
        self.settings.setValue("trace/init_args", init_args_text)
        self.settings.setValue("trace/input_shapes", self.shapes.toPlainText())
        self.settings.setValue("trace/input_dtypes", self.dtypes.toPlainText())
        self.settings.setValue("trace/device", request.device)
        self.request = request
        self.accept()


class TraceWorker(QObject):
    finished = Signal(str, str)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, request: TraceRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            adapted = render_supported_project(self.request.path)
            if adapted is not None:
                html, model_name = adapted
                self.finished.emit(html, model_name)
                return
            model = load_model(self.request.path, self.request.symbol, self.request.init_args)
            model.to(self.request.device)
            inputs = make_inputs(self.request.shapes, self.request.dtypes, self.request.device)
            graph = trace_model(model, inputs)
            self.finished.emit(render_html(graph), graph["name"])
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.completed.emit()
            # Stop the event loop from inside the worker thread. PySide 6.12
            # does not reliably adapt result signals with arguments to the
            # zero-argument QThread.quit slot across threads.
            QThread.currentThread().quit()


class DesktopPage(QWebEnginePage):
    """Handle private links from the desktop welcome screen."""

    recentModelRequested = Signal(str)
    browseRequested = Signal()

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if url.host() == "modviz.local":
            if url.path() == "/open":
                path = QUrlQuery(url).queryItemValue("path")
                if path:
                    self.recentModelRequested.emit(path)
            elif url.path() == "/browse":
                self.browseRequested.emit()
            return False
        return super().acceptNavigationRequest(url, navigation_type, is_main_frame)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("ModViz", "ModViz Desktop")
        self.current_html = ""
        self.current_request: TraceRequest | None = None
        self.current_path: Path | None = None
        self._thread: QThread | None = None
        self._worker: TraceWorker | None = None
        self.setWindowTitle("ModViz — PyTorch Network Visualizer")
        self.setWindowIcon(application_icon())
        self.resize(1500, 920)

        self.web = QWebEngineView(self)
        page = DesktopPage(self.web)
        page.recentModelRequested.connect(self._open_recent_model)
        page.browseRequested.connect(lambda: QTimer.singleShot(0, self.open_model))
        self.web.setPage(page)
        self.setCentralWidget(self.web)
        self.web.page().profile().downloadRequested.connect(self._save_download)
        self._create_actions()
        self._show_welcome()
        self.statusBar().showMessage("Open a trusted Python model file to begin")

    def _create_actions(self) -> None:
        toolbar = QToolBar("Application", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open model…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_model)
        toolbar.addAction(open_action)

        self.reload_action = QAction("Reload", self)
        self.reload_action.setShortcut(QKeySequence.Refresh)
        self.reload_action.setEnabled(False)
        self.reload_action.triggered.connect(self.reload_model)
        toolbar.addAction(self.reload_action)

        self.save_action = QAction("Save…", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_diagram)
        toolbar.addAction(self.save_action)

        toolbar.addSeparator()
        help_label = QLabel("Edit, group, compact, toggle losses, and export inside the diagram below.")
        toolbar.addWidget(help_label)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(self.reload_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _show_welcome(self) -> None:
        self.web.setHtml(welcome_html(self._recent_models()))

    def _recent_models(self) -> list[Path]:
        return recent_models(self.settings)

    def _remember_recent(self, path: Path) -> None:
        remember_recent_model(self.settings, path)

    @Slot(str)
    def _open_recent_model(self, filename: str) -> None:
        QTimer.singleShot(0, lambda: self.open_path(Path(filename)))

    @Slot()
    def open_model(self) -> None:
        start = str(self.settings.value("files/last_directory", str(Path.home())))
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open model or ModViz diagram",
            start,
            "ModViz files (*.py *.html);;Python models (*.py);;ModViz diagrams (*.html)",
        )
        if not filename:
            return
        path = Path(filename).resolve()
        self.settings.setValue("files/last_directory", str(path.parent))
        if path.suffix.lower() in {".html", ".htm"}:
            self._open_html(path)
            return
        if vessel_project_root(path) is not None:
            self._remember_recent(path)
            self._start_trace(TraceRequest(path, None, {}, [], [], "cpu"))
            return
        dialog = ModelSetupDialog(path, self.settings, self)
        if dialog.exec() == QDialog.Accepted and dialog.request:
            self._remember_recent(path)
            self._start_trace(dialog.request)

    @Slot()
    def reload_model(self) -> None:
        if self.current_path and self.current_path.suffix.lower() in {".html", ".htm"}:
            self._open_html(self.current_path)
            return
        if self.current_request:
            self._start_trace(self.current_request)

    def open_path(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            QMessageBox.warning(self, "Model not found", f"The model file no longer exists:\n{path}")
            self._show_welcome()
            return
        if path.suffix.lower() in {".html", ".htm"}:
            self._open_html(path)
            return
        if vessel_project_root(path) is not None:
            self._remember_recent(path)
            self._start_trace(TraceRequest(path, None, {}, [], [], "cpu"))
            return
        dialog = ModelSetupDialog(path, self.settings, self)
        if dialog.exec() == QDialog.Accepted and dialog.request:
            self._remember_recent(path)
            self._start_trace(dialog.request)

    def _open_html(self, path: Path) -> None:
        try:
            document = path.read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Could not open diagram", str(exc))
            return
        if "exportLayoutState" not in document or 'id="svg"' not in document:
            QMessageBox.warning(self, "Not a ModViz diagram", "Choose an HTML file previously saved by ModViz.")
            return
        self.current_request = None
        self.current_path = path
        self.current_html = document
        self._remember_recent(path)
        self.web.setHtml(document, QUrl.fromLocalFile(str(path.parent) + "/"))
        self.setWindowTitle(f"{path.stem} — ModViz Desktop")
        self.reload_action.setEnabled(True)
        self.save_action.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {path}")

    def _start_trace(self, request: TraceRequest) -> None:
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    QMessageBox.information(self, "Trace in progress", "Wait for the current model trace to finish.")
                    return
            except RuntimeError:
                # Qt may already have deleted the C++ QThread while the Python
                # wrapper is still referenced. Treat that wrapper as finished.
                self._thread = None
                self._worker = None
        self.current_request = request
        self.current_path = request.path
        self.reload_action.setEnabled(False)
        self.save_action.setEnabled(False)
        self.statusBar().showMessage(f"Tracing {request.path.name}…")

        thread = QThread(self)
        worker = TraceWorker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._trace_finished)
        worker.failed.connect(self._trace_failed)
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(self._trace_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot()
    def _trace_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    @Slot(str, str)
    def _trace_finished(self, html: str, model_name: str) -> None:
        self.current_html = html
        base = QUrl.fromLocalFile(str(self.current_request.path.parent) + "/") if self.current_request else QUrl()
        self.web.setHtml(html, base)
        self.setWindowTitle(f"{model_name} — ModViz Desktop")
        self.reload_action.setEnabled(True)
        self.save_action.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {self.current_request.path}")

    @Slot(str)
    def _trace_failed(self, details: str) -> None:
        self.reload_action.setEnabled(self.current_request is not None)
        self.statusBar().showMessage("Model trace failed")
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle("Could not visualize model")
        dialog.setText("The model could not be loaded or traced.")
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def save_diagram(self) -> None:
        if not self.current_html:
            return
        source = self.current_path or (self.current_request.path if self.current_request else Path("network"))
        suggested = source.name if source.suffix.lower() == ".html" else source.stem + "_flow.html"
        filename, _ = QFileDialog.getSaveFileName(self, "Save ModViz diagram", suggested, "ModViz HTML (*.html)")
        if not filename:
            return
        target = Path(filename).expanduser().resolve()
        if target.suffix.lower() != ".html":
            target = target.with_suffix(".html")
        self.web.page().runJavaScript(
            "exportLayoutState()",
            lambda layout, path=target: self._save_diagram_result(path, layout),
        )

    def _save_diagram_result(self, target: Path, layout) -> None:
        if not isinstance(layout, str) or not layout:
            QMessageBox.warning(self, "Could not save diagram", "The editor did not return a valid layout.")
            return
        try:
            document = embed_layout_html(self.current_html, layout)
            target.write_text(document, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save diagram", str(exc))
            return
        self.current_html = document
        self.current_path = target
        self.current_request = None
        self._remember_recent(target)
        self.setWindowTitle(f"{target.stem} — ModViz Desktop")
        self.statusBar().showMessage(f"Saved {target}", 5000)

    @Slot(object)
    def _save_download(self, download: QWebEngineDownloadRequest) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export diagram", download.suggestedFileName())
        if not filename:
            download.cancel()
            return
        target = Path(filename).resolve()
        download.setDownloadDirectory(str(target.parent))
        download.setDownloadFileName(target.name)
        download.accept()
        self.statusBar().showMessage(f"Exporting {target.name}…", 5000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the standalone ModViz desktop application")
    parser.add_argument("file", nargs="?", type=Path, help="Optional Python model or saved ModViz HTML file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("ModViz Desktop")
    app.setApplicationDisplayName("ModViz")
    app.setOrganizationName("ModViz")
    app.setWindowIcon(application_icon())
    window = MainWindow()
    window.show()
    if args.file:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: window.open_path(args.file))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
