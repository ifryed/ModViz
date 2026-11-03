"""Rasterize a geometry-test SVG for inspection, without running a browser."""
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

app = QGuiApplication(["geometry-preview", "-platform", "offscreen"])
renderer = QSvgRenderer(sys.argv[1])
size = renderer.defaultSize()
size.scale(2400, 1600, Qt.AspectRatioMode.KeepAspectRatio)
image = QImage(size, QImage.Format.Format_ARGB32)
image.fill(Qt.GlobalColor.white)
painter = QPainter(image)
renderer.render(painter)
painter.end()
if not image.save(sys.argv[2]):
    raise RuntimeError("Could not save preview")
