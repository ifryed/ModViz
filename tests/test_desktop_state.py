import tempfile
import unittest
from pathlib import Path

from torch_flow_viz.desktop_state import (
    embed_layout_html,
    recent_models,
    remember_recent_model,
    welcome_html,
)


class FakeSettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class DesktopStateTests(unittest.TestCase):
    def test_recent_models_are_ordered_deduplicated_and_stale_entries_removed(self):
        settings = FakeSettings()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.touch()
            second.touch()
            settings.setValue("files/recent_models", [str(first), str(Path(directory) / "gone.py")])
            remember_recent_model(settings, second)
            remember_recent_model(settings, first)
            self.assertEqual(recent_models(settings), [first.resolve(), second.resolve()])

    def test_welcome_links_are_escaped(self):
        first = Path("/tmp/A & B.py")
        page = welcome_html([first])
        self.assertIn("Recent files", page)
        self.assertIn("A &amp; B.py", page)
        self.assertIn("https://modviz.local/open?path=%2Ftmp%2FA%20%26%20B.py", page)

    def test_layout_is_embedded_safely_and_replaced_on_resave(self):
        document = "<html><script>function importLayoutState(value){return value}</script></html>"
        saved = embed_layout_html(document, '{"name":"</script>"}')
        self.assertEqual(saved.count("MODVIZ_SAVED_LAYOUT_START"), 1)
        self.assertIn(r"\u003c/script>", saved)
        self.assertIn("setTimeout(()=>importLayoutState", saved)

        resaved = embed_layout_html(saved, '{"name":"second"}')
        self.assertEqual(resaved.count("MODVIZ_SAVED_LAYOUT_START"), 1)
        self.assertNotIn(r"\u003c/script>", resaved)
        self.assertIn(r'\"second\"', resaved)
