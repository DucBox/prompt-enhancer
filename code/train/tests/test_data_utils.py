"""Test thuần logic cho code/train/data_utils.py -- không cần torch/transformers/GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_utils  # noqa: E402

PHOTO_TARGET = {
    "high_level_description": "A ceramic Ấm tích teapot on a tray.",
    "style_description": {"aesthetics": "clean", "lighting": "soft", "photo": "close-up",
                          "medium": "photograph"},
    "compositional_deconstruction": {
        "background": "grey studio backdrop",
        "elements": [{"type": "obj", "desc": "Ấm tích teapot"},
                     {"type": "text", "text": "PHỐ CỔ", "desc": "red sign"}],
    },
}
ART_TARGET = {
    "high_level_description": "A golden chim Lạc render.",
    "style_description": {"aesthetics": "minimalist", "lighting": "studio", "medium": "3d_render",
                          "art_style": "polished gold"},
    "compositional_deconstruction": {"background": "white", "elements": [{"type": "obj", "desc": "chim Lạc"}]},
}


def minify(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class TestReadJsonlRows(unittest.TestCase):

    def write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write("\n")  # dòng trống cuối file phải được bỏ qua
        f.close()
        return f.name

    def test_mixed_schemas_keep_exact_keys_and_order(self):
        """Hồi quy: load_dataset("json") chèn art_style/photo/text = null vào mọi dòng."""
        path = self.write([{"mode": "short", "target_json": PHOTO_TARGET},
                           {"mode": "long", "target_json": ART_TARGET}])
        rows = data_utils.read_jsonl_rows(path)
        self.assertEqual(len(rows), 2)
        for row, gold in zip(rows, (PHOTO_TARGET, ART_TARGET)):
            out = data_utils.target_to_minified_json(row["target_json"], validate=True,
                                                     allow_bbox_palette=False)
            self.assertEqual(out, minify(gold))
            self.assertNotIn("null", out)

    def test_invalid_line_reports_line_number(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        f.write('{"a": 1}\nkhông phải json\n')
        f.close()
        with self.assertRaisesRegex(ValueError, ":2:"):
            data_utils.read_jsonl_rows(f.name)


class TestTargetValidation(unittest.TestCase):

    def check(self, target, allow=False):
        return data_utils.target_to_minified_json(target, validate=True, allow_bbox_palette=allow)

    def test_keeps_vietnamese_literal(self):
        self.assertIn("Ấm tích", self.check(PHOTO_TARGET))

    def test_rejects_null_keys(self):
        bad = json.loads(json.dumps(PHOTO_TARGET))
        bad["style_description"]["art_style"] = None
        with self.assertRaisesRegex(ValueError, "null"):
            self.check(bad)

    def test_rejects_id_from_step2c(self):
        bad = json.loads(json.dumps(PHOTO_TARGET))
        bad["compositional_deconstruction"]["elements"][0] = {"id": 0, "type": "obj", "desc": "x"}
        with self.assertRaisesRegex(ValueError, "step2d_final"):
            self.check(bad)
        with self.assertRaisesRegex(ValueError, "step2d_final"):
            self.check(bad, allow=True)

    def test_rejects_bbox_unless_allowed(self):
        bad = json.loads(json.dumps(PHOTO_TARGET))
        bad["compositional_deconstruction"]["elements"][0]["bbox"] = [0, 0, 10, 10]
        with self.assertRaises(ValueError):
            self.check(bad)
        self.assertIn("bbox", self.check(bad, allow=True))

    def test_rejects_missing_top_level(self):
        bad = {k: v for k, v in PHOTO_TARGET.items() if k != "style_description"}
        with self.assertRaises(ValueError):
            self.check(bad)


if __name__ == "__main__":
    unittest.main()
