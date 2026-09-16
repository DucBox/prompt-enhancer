"""Test phần chuẩn bị dữ liệu: in tiến độ, lọc dòng hỏng / quá dài, mask nhãn đúng prefix.

Không cần GPU. Máy dev chưa cài torch/datasets/transformers thì test tự tạo bản giả tối thiểu
để import được module train (chỉ dùng các hàm thuần Python trong đó).
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

TRAIN_DIR = Path(__file__).resolve().parents[1]


def _ensure_stub(name: str, build) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = build()


_ensure_stub("torch", lambda: types.SimpleNamespace(
    __name__="torch", Tensor=type("Tensor", (), {}), tensor=lambda *a, **k: None,
    long=None, cuda=types.SimpleNamespace(is_available=lambda: False),
))


def _stub_datasets():
    mod = types.ModuleType("datasets")
    mod.Dataset = type("Dataset", (), {"from_list": staticmethod(lambda rows: rows)})
    return mod


def _stub_transformers():
    mod = types.ModuleType("transformers")
    for name in ("Trainer", "TrainerCallback", "TrainingArguments"):
        setattr(mod, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    mod.set_seed = lambda seed: None
    return mod


_ensure_stub("datasets", _stub_datasets)
_ensure_stub("transformers", _stub_transformers)

sys.argv = ["train_prompt_enhancer_qwen36.py", "--backend", "hf"]  # tránh import unsloth
_spec = importlib.util.spec_from_file_location(
    "train_module", TRAIN_DIR / "train_prompt_enhancer_qwen36.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["train_module"] = T
_spec.loader.exec_module(T)

# Không đụng tới datasets thật: ở đây chỉ cần list các dòng đã encode.
T.Dataset = type("FakeDataset", (), {"from_list": staticmethod(lambda rows: list(rows))})

SYS = "SYS"  # ép dùng 1 system prompt ngắn -> độ dài trong test dễ kiểm soát


class FakeTokenizer:
    """Chat template giả, tokenize theo ký tự -> prefix luôn là tiền tố của bản đầy đủ."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False, preserve_thinking=False):
        text = "".join("<{}>{}".format(m["role"], m["content"]) for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return {"input_ids": [ord(c) for c in text]} if tokenize else text


def make_args(**over):
    args = SimpleNamespace(
        prompt_field="user_prompt", cot_field="cot", target_field="target_json",
        detail_level_field="mode", mode="no_cot", max_seq_length=10_000,
        skip_json_validation=True, allow_bbox_palette=False, prepare_log_every=0,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def make_rows(n=5):
    return [{"user_prompt": f"prompt {i}", "mode": ["short", "medium", "long"][i % 3],
             "target_json": {"high_level_description": f"caption {i}"}} for i in range(n)]


def write_jsonl(tmp: Path, rows):
    path = tmp / "data.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


class TestEncode(unittest.TestCase):

    def test_prepare_matches_encode_record_row_by_row(self):
        tok, args, rows = FakeTokenizer(), make_args(), make_rows(5)
        with tempfile.TemporaryDirectory() as d:
            path = write_jsonl(Path(d), rows)
            with redirect_stdout(io.StringIO()):
                prepared = T.prepare_dataset(path, tok, args, SYS)
        expected = [T.encode_record(r, tok, args, SYS) for r in rows]
        for got, want in zip(prepared, expected):
            want.pop("_keep"), want.pop("_length")
            self.assertEqual(got, want)

    def test_labels_mask_exactly_the_prefix(self):
        tok, args, row = FakeTokenizer(), make_args(), make_rows(1)[0]
        item = T.encode_record(row, tok, args, SYS)
        prefix_len = len(tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": row["user_prompt"]}],
            tokenize=True, add_generation_prompt=True)["input_ids"])
        self.assertEqual(sum(1 for lab in item["labels"] if lab == -100), prefix_len)
        self.assertEqual(item["attention_mask"], [1] * len(item["input_ids"]))


class TestFiltering(unittest.TestCase):

    def test_skips_invalid_rows_and_too_long_rows(self):
        rows = make_rows(4)
        rows[1]["user_prompt"] = "   "                                     # dòng hỏng
        rows[2]["target_json"] = {"high_level_description": "x" * 5_000}   # quá dài
        tok, args = FakeTokenizer(), make_args(max_seq_length=200)
        with tempfile.TemporaryDirectory() as d:
            path = write_jsonl(Path(d), rows)
            out = io.StringIO()
            with redirect_stdout(out):
                prepared = T.prepare_dataset(path, tok, args, SYS)
        self.assertEqual(len(prepared), 2)
        self.assertIn("invalid=1", out.getvalue())
        self.assertIn("over_max_seq=1", out.getvalue())

    def test_all_rows_invalid_raises(self):
        rows = [{"user_prompt": "", "mode": "short", "target_json": {}}]
        with tempfile.TemporaryDirectory() as d:
            path = write_jsonl(Path(d), rows)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                T.prepare_dataset(path, FakeTokenizer(), make_args(), SYS)


class TestProgress(unittest.TestCase):

    def test_prints_on_interval_and_at_the_end(self):
        out = io.StringIO()
        with redirect_stdout(out):
            p = T.Progress("encode", 10, every=4)
            for done in range(1, 11):
                p.update(done)
        lines = [x for x in out.getvalue().splitlines() if x.startswith("[prepare]")]
        self.assertEqual(len(lines), 3)          # tại 4, 8 và 10
        self.assertIn("10/10 (100%)", lines[-1])
        self.assertIn("dòng/s", lines[-1])

    def test_counts_every_row_including_skipped_ones(self):
        rows = make_rows(6)
        rows[0]["user_prompt"] = ""              # dòng hỏng vẫn phải được tính vào tiến độ
        tok, args = FakeTokenizer(), make_args(prepare_log_every=3)
        with tempfile.TemporaryDirectory() as d:
            path = write_jsonl(Path(d), rows)
            out = io.StringIO()
            with redirect_stdout(out):
                T.prepare_dataset(path, tok, args, SYS)
        lines = [x for x in out.getvalue().splitlines() if x.startswith("[prepare] encode:")]
        self.assertIn("6/6 (100%)", lines[-1])

    def test_quiet_on_non_zero_rank(self):
        os.environ["RANK"] = "3"
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                T.Progress("encode", 4, every=1).update(4)
            self.assertEqual(out.getvalue(), "")
        finally:
            os.environ["RANK"] = "0"


if __name__ == "__main__":
    unittest.main()
