#!/usr/bin/env python3
"""Kiểm tra môi trường có đủ để chạy train_prompt_enhancer_qwen36.py chưa, thiếu thì in lệnh cài.

KHÔNG so số phiên bản cứng -- kiểm tra thẳng thứ code train cần (vd transformers có biết
kiến trúc `qwen3_5` của Qwen3.6 không), vì số phiên bản ghi trong config của model không
đáng tin (config ghi 4.57.1 nhưng 4.57.x chưa có qwen3_5).

    python3 check_env.py                         # backend unsloth (mặc định)
    python3 check_env.py --backend hf
    # thêm: tải tokenizer + encode vài dòng dữ liệu thật bằng đúng hàm của script train
    python3 check_env.py --data_file ../method_1/outputs/test_5_new/step2d_final/train.jsonl

Mã thoát: 0 nếu đủ để train, 1 nếu còn thiếu thứ bắt buộc.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    pip: List[str] = field(default_factory=list)  # gói cần cài/nâng cấp nếu không OK


# Gói bắt buộc: (tên module import, tên gói pip)
COMMON_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("jinja2", "jinja2"),
    ("huggingface_hub", "huggingface_hub"),
]
BACKEND_PACKAGES = {
    "unsloth": [("unsloth", "unsloth"), ("unsloth_zoo", "unsloth_zoo")],
    "hf": [],
}
# Kernel cho Gated DeltaNet (linear attention). Thiếu thì transformers rơi về bản torch
# thuần -- vẫn chạy nhưng chậm và tốn bộ nhớ hơn nhiều.
FAST_KERNELS = [("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm tra môi trường train Prompt Enhancer")
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    p.add_argument("--model_name", default="unsloth/Qwen3.6-27B")
    p.add_argument("--data_file", default=None,
                   help="JSONL step2d_final: tải tokenizer và encode thử vài dòng (không cần GPU)")
    p.add_argument("--n_rows", type=int, default=20, help="Số dòng encode thử với --data_file")
    p.add_argument("--min_vram_gb", type=float, default=40.0,
                   help="VRAM tối thiểu mỗi GPU để cảnh báo (QLoRA 27B ~ A100 40GB)")
    return p.parse_args()


def version_of(module) -> str:
    return str(getattr(module, "__version__", "?"))


def check_python() -> Check:
    v = sys.version_info
    text = "{}.{}.{}".format(v.major, v.minor, v.micro)
    if v >= (3, 10):
        return Check("python", OK, text)
    return Check("python", FAIL, text + " -- cần >= 3.10 (transformers bản mới không hỗ trợ 3.9)")


def check_package(module_name: str, pip_name: str, required: bool = True) -> Check:
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 - lỗi import nào cũng phải báo, không làm sập script
        status = FAIL if required else WARN
        reason = "chưa cài" if isinstance(e, ModuleNotFoundError) else "import lỗi: {}".format(e)
        return Check(pip_name, status, reason, [pip_name])
    return Check(pip_name, OK, version_of(module))


def check_transformers_capabilities(backend: str) -> List[Check]:
    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:  # noqa: BLE001
        return []
    checks = []
    if "qwen3_5" in CONFIG_MAPPING_NAMES:
        checks.append(Check("transformers: kiến trúc qwen3_5", OK))
    else:
        checks.append(Check("transformers: kiến trúc qwen3_5", FAIL,
                            "bản {} chưa có Qwen3.5/3.6 -- nâng cấp".format(transformers.__version__),
                            ["transformers"]))
    if backend == "hf":
        has = hasattr(transformers, "AutoModelForMultimodalLM")
        checks.append(Check("transformers: AutoModelForMultimodalLM", OK if has else FAIL,
                            "" if has else "cần cho --backend hf", [] if has else ["transformers"]))
    try:
        import inspect
        params = inspect.signature(transformers.TrainingArguments.__init__).parameters
        ok = "eval_strategy" in params
        checks.append(Check("transformers: TrainingArguments(eval_strategy)", OK if ok else FAIL,
                            "" if ok else "bản quá cũ", [] if ok else ["transformers"]))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("transformers: TrainingArguments", FAIL, str(e), ["transformers"]))
    return checks


def check_gpu(min_vram_gb: float) -> List[Check]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if not torch.cuda.is_available():
        return [Check("CUDA", FAIL, "torch {} không thấy GPU (torch bản CPU, hoặc sai driver/CUDA)".format(
            torch.__version__))]
    checks = [Check("CUDA", OK, "torch {} | CUDA {} | {} GPU".format(
        torch.__version__, torch.version.cuda, torch.cuda.device_count()))]
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        vram = prop.total_memory / 1024 ** 3
        status = OK if vram >= min_vram_gb else WARN
        checks.append(Check("GPU {}".format(i), status, "{} | {:.0f} GB | compute {}.{}".format(
            prop.name, vram, prop.major, prop.minor)))
    bf16 = torch.cuda.is_bf16_supported()
    checks.append(Check("bf16", OK if bf16 else FAIL,
                        "" if bf16 else "script train bật bf16/tf32 -- cần GPU Ampere trở lên"))
    return checks


def check_project_imports() -> List[Check]:
    sys.path.insert(0, str(HERE))
    checks = []
    for module_name in ("prompts", "data_utils", "train_prompt_enhancer_qwen36"):
        try:
            importlib.import_module(module_name)
            checks.append(Check("import " + module_name, OK))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("import " + module_name, FAIL, "{}: {}".format(type(e).__name__, e)))
    return checks


def check_data_encoding(model_name: str, data_file: str, n_rows: int) -> List[Check]:
    """Tải tokenizer (vài MB, không tải trọng số) và encode thử bằng đúng encode_record."""
    try:
        import train_prompt_enhancer_qwen36 as T
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        return [Check("encode dữ liệu", FAIL, "không import được script train: {}".format(e))]
    try:
        tok = T.get_text_tokenizer(AutoTokenizer.from_pretrained(model_name))
    except Exception as e:  # noqa: BLE001
        return [Check("tokenizer " + model_name, FAIL, str(e)[:300])]
    args = argparse.Namespace(prompt_field="user_prompt", cot_field="cot", target_field="target_json",
                              detail_level_field="mode", mode="no_cot", skip_json_validation=False,
                              allow_bbox_palette=False, max_seq_length=4096)
    rows = T.read_jsonl_rows(data_file)[:n_rows]
    errors, lengths = [], []
    for row in rows:
        try:
            lengths.append(T.encode_record(row, tok, args, None)["_length"])
        except Exception as e:  # noqa: BLE001
            errors.append("{}: {}".format(row.get("id", "?"), e))
    checks = [Check("tokenizer " + model_name, OK)]
    if errors:
        checks.append(Check("encode {} dòng".format(len(rows)), FAIL,
                            "{} lỗi, vd {}".format(len(errors), errors[0][:300])))
    else:
        checks.append(Check("encode {} dòng".format(len(rows)), OK,
                            "token dài nhất {} (max_seq_length 4096)".format(max(lengths, default=0))))
    return checks


def install_commands(checks: List[Check]) -> List[str]:
    """Gom gói cần cài thành lệnh pip. torch/bitsandbytes tách riêng vì phụ thuộc CUDA."""
    wanted: Dict[str, None] = {}
    for c in checks:
        if c.status != OK:
            for name in c.pip:
                wanted.setdefault(name, None)
    if not wanted:
        return []
    cuda_bound = [n for n in wanted if n in ("torch", "bitsandbytes")]
    kernels = [n for n in wanted if n in dict((p, m) for m, p in FAST_KERNELS)]
    rest = [n for n in wanted if n not in cuda_bound and n not in kernels]
    commands = []
    if cuda_bound:
        commands.append("pip install -U {}   # phụ thuộc CUDA -- torch lấy đúng bản theo CUDA của máy: "
                        "https://pytorch.org/get-started/locally/".format(" ".join(cuda_bound)))
    if rest:
        commands.append("pip install -U " + " ".join(rest))
    if kernels:
        commands.append("pip install -U {} --no-build-isolation   # tuỳ chọn, tăng tốc Gated DeltaNet".format(
            " ".join(kernels)))
    return commands


def run_all(args: argparse.Namespace) -> List[Check]:
    checks: List[Check] = [check_python()]
    packages = COMMON_PACKAGES + BACKEND_PACKAGES[args.backend]
    # unsloth phải import TRƯỚC transformers/peft để nó vá được -- giữ đúng thứ tự như lúc train.
    if args.backend == "unsloth":
        packages = BACKEND_PACKAGES["unsloth"][:1] + [p for p in packages if p[0] != "unsloth"]
    checks += [check_package(m, p) for m, p in packages]
    checks += [check_package(m, p, required=False) for m, p in FAST_KERNELS]
    checks += check_transformers_capabilities(args.backend)
    checks += check_gpu(args.min_vram_gb)
    checks += check_project_imports()
    if args.data_file:
        checks += check_data_encoding(args.model_name, args.data_file, args.n_rows)
    return checks


def report(checks: List[Check], printer: Callable[[str], None] = print) -> bool:
    width = max(len(c.name) for c in checks)
    for c in checks:
        printer("  [{:<4}] {}  {}".format(c.status, c.name.ljust(width), c.detail))
    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    printer("")
    commands = install_commands(checks)
    if commands:
        printer("Cần cài thêm:")
        for cmd in commands:
            printer("  " + cmd)
        printer("")
    if failed:
        printer("CHƯA ĐỦ để train: {} mục FAIL, {} cảnh báo.".format(len(failed), len(warned)))
        return False
    printer("ĐỦ để train ({} cảnh báo).".format(len(warned)))
    return True


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    print("Kiểm tra môi trường train  (backend={}, model={})\n".format(args.backend, args.model_name))
    return 0 if report(run_all(args)) else 1


def parse_args_from(argv: List[str]) -> argparse.Namespace:
    old = sys.argv
    try:
        sys.argv = [old[0]] + argv
        return parse_args()
    finally:
        sys.argv = old


if __name__ == "__main__":
    sys.exit(main())
