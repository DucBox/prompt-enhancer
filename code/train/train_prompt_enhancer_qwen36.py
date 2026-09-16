#!/usr/bin/env python3
"""SFT Qwen3.6-27B as an Ideogram-4 prompt enhancer.

Supports:
  * with CoT:   train native Qwen3.6 <think> reasoning + final JSON
  * without CoT: train direct JSON only; Qwen3.6's chat template supplies an empty think block
  * 1 GPU: Unsloth FastModel QLoRA (recommended OSS path)
  * multi GPU: Transformers + PEFT QLoRA under torchrun/DDP

Expected JSONL (field names are configurable; matches code/method_1/step2d_finalize.py output,
i.e. <out_root>/step2d_final/{train,val}.jsonl -- NOT step2c_split, whose targets still carry `id`):
  {"user_prompt":"...", "mode":"short|medium|long", "cot":"...", "target_json": {...}}

The target should already be your normalized Prompt-Enhancer target (for this project:
Ideogram-4-like JSON without id/bbox/color_palette fields).

Data is read line by line with json.loads (data_utils.read_jsonl_rows), NOT
datasets.load_dataset("json"): Arrow would unify target_json into one struct and insert
null keys (art_style/photo/text) into every row.

"mode" (short/medium/long) selects WHICH SYSTEM PROMPT is used for that row -- see
prompts.py. Each level has its own distinct system prompt (not a shared prompt with
a one-line tag), because the model's job genuinely differs per level: short needs to
invent detail, long needs to faithfully structure what's already given. The same
per-level selection must happen at inference time; there is no way to train this
without the caller knowing which level it is serving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _wants_unsloth(argv: List[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            return argv[i + 1] == "unsloth"
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1] == "unsloth"
    return True  # --backend mặc định là unsloth


def _argv_value(argv: List[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


# --model_name là folder local -> tự bật chế độ offline HF (server không ra mạng: không bật thì
# from_pretrained cố gọi Hub và treo rất lâu). Phải đặt TRƯỚC khi import transformers /
# huggingface_hub vì chúng đọc biến này lúc import. Người dùng đã tự đặt thì giữ nguyên.
if __name__ == "__main__":
    _model_arg = _argv_value(sys.argv[1:], "--model_name")
    if _model_arg and os.path.isdir(_model_arg):
        for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.setdefault(_var, "1")


# ---------------------------------------------------------------------------------------
# DEBUG: bật bằng --debug hoặc PE_DEBUG=1. Phải quyết định TRƯỚC khi import torch (argparse
# chạy sau import) nên đọc thẳng sys.argv. Mỗi dòng log có rank + thời gian từ lúc khởi động +
# thời gian của bước, flush ngay (không bị buffer). Ngoài ra faulthandler tự in stack của MỌI
# luồng mỗi PE_DEBUG_STACK_EVERY giây (mặc định 300, 0 = tắt) -> treo ở đâu thì thấy ngay.
# ---------------------------------------------------------------------------------------
import contextlib  # noqa: E402
import faulthandler  # noqa: E402
import time  # noqa: E402

DEBUG = "--debug" in sys.argv[1:] or os.environ.get("PE_DEBUG", "") == "1"
_T0 = time.time()


def dbg(msg: str) -> None:
    if DEBUG:
        rank = os.environ.get("RANK", "0")
        print(f"[debug r{rank} +{time.time() - _T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def dbg_step(name: str):
    """Log BẮT ĐẦU / XONG (kèm thời gian) quanh một bước; không làm gì khi tắt debug."""
    if not DEBUG:
        yield
        return
    start = time.time()
    dbg(f"-> {name} ...")
    yield
    dbg(f"<- {name} xong ({time.time() - start:.1f}s)")


if DEBUG:
    _stack_every = int(os.environ.get("PE_DEBUG_STACK_EVERY", "300"))
    faulthandler.enable(file=sys.stderr)
    if _stack_every > 0:
        faulthandler.dump_traceback_later(_stack_every, repeat=True, file=sys.stderr)
    dbg(f"debug bật; python={sys.version.split()[0]} pid={os.getpid()} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} stack mỗi {_stack_every}s")


# Unsloth phải được import TRƯỚC transformers/peft thì mới vá được kernel; import muộn (trong
# load_unsloth_model) chỉ cảnh báo và chạy chậm hơn. Chỉ làm khi chạy thẳng script với backend
# unsloth -- check_env.py import module này thì không kéo unsloth vào.
if __name__ == "__main__" and _wants_unsloth(sys.argv[1:]):
    with dbg_step("import unsloth"):
        import unsloth  # noqa: F401,E402

with dbg_step("import torch"):
    import torch  # noqa: E402
with dbg_step("import datasets"):
    from datasets import Dataset  # noqa: E402
with dbg_step("import transformers"):
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with dbg_step("import data_utils, prompts"):
    from data_utils import read_jsonl_rows, target_to_minified_json  # noqa: E402
    from prompts import LEVELS, build_system_prompt  # noqa: E402


# Qwen3.6 hybrid language stack: standard attention + GatedDeltaNet + MLP.
# For the HF/PEFT multi-GPU fallback we discover exact language-module paths ending
# in these projection names, avoiding accidental LoRA insertion into the vision tower/MTP.
QWEN36_LORA_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_q", "in_proj_k", "in_proj_v",
    "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True, help="JSONL training file")
    p.add_argument("--eval_file", default=None, help="Optional JSONL validation file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="unsloth/Qwen3.6-27B")

    p.add_argument("--mode", choices=["cot", "no_cot"], required=True)
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth",
                   help="unsloth=single-GPU OSS path; hf=torchrun/DDP multi-GPU fallback")

    p.add_argument("--prompt_field", default="user_prompt")
    p.add_argument("--cot_field", default="cot")
    p.add_argument("--target_field", default="target_json")
    p.add_argument("--detail_level_field", default="mode",
                   help="Trường trong JSONL ghi short/medium/long (khớp field 'mode' "
                        "do step2c_split.py sinh ra). Mỗi mức dùng MỘT system prompt "
                        "riêng (xem prompts.py) -- không phải một tag chung.")
    p.add_argument("--system_prompt_file", default=None,
                   help="Nâng cao/debug: ép DÙNG CHUNG một system prompt cho MỌI dòng, "
                        "bỏ qua lựa chọn theo detail_level_field. Mặc định (để trống) "
                        "sẽ tự chọn system prompt đúng theo mức của từng dòng.")

    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--use_rslora", action="store_true")

    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--global_batch_size", type=int, default=16,
                   help="Used to auto-compute gradient accumulation from WORLD_SIZE")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", default="cosine")

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", default="none", help="none or wandb")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prepare_log_every", type=int, default=2000,
                   help="In tiến độ chuẩn bị dữ liệu mỗi N dòng")

    p.add_argument("--allow_bbox_palette", action="store_true",
                   help="By default, reject targets containing bbox/color_palette to match this PE project")
    p.add_argument("--skip_json_validation", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="In log từng bước (import, load model, data, train) kèm thời gian + VRAM, "
                        "và tự in stack mọi luồng mỗi PE_DEBUG_STACK_EVERY giây. Tương đương PE_DEBUG=1.")
    return p.parse_args()


def distributed_info() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    return world_size, local_rank, rank


def load_system_prompt_override(path: Optional[str]) -> Optional[str]:
    """None nghĩa là KHÔNG override -- mỗi dòng sẽ tự chọn system prompt theo mức
    (xem resolve_system_prompt). Chỉ trả về non-None khi người dùng chủ động ép
    dùng chung một prompt qua --system_prompt_file (trường hợp debug/nâng cao)."""
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def resolve_system_prompt(ex: Dict[str, Any], args: argparse.Namespace,
                          override: Optional[str]) -> str:
    if override is not None:
        return override
    level = str(ex.get(args.detail_level_field, "")).strip().lower()
    if level not in LEVELS:
        raise ValueError(
            "thiếu/sai trường '{}' (giá trị: {!r}) -- cần một trong {}; hoặc dùng "
            "--system_prompt_file để ép dùng chung một prompt".format(
                args.detail_level_field, level, LEVELS)
        )
    return build_system_prompt(level)


def get_text_tokenizer(processor_or_tokenizer: Any):
    tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_messages(system_prompt: str, user_prompt: str, target_json: str,
                   mode: str, cot: str) -> List[Dict[str, Any]]:
    assistant: Dict[str, Any] = {"role": "assistant", "content": target_json}
    if mode == "cot":
        assistant["reasoning_content"] = cot
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        assistant,
    ]


def _as_ids(x: Any) -> List[int]:
    # tokenizer.apply_chat_template(tokenize=True) normally returns List[int].
    # Newer transformers return a BatchEncoding (a UserDict, NOT a dict subclass):
    # list() on it yields the KEY NAMES ("input_ids", "attention_mask"), which silently
    # passes the prefix check and crashes later in the collator. Handle any mapping.
    if isinstance(x, Mapping) or hasattr(x, "keys"):
        x = x["input_ids"]
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    ids = list(x)
    if not all(isinstance(i, int) for i in ids):
        raise TypeError(f"apply_chat_template returned non-token ids: {type(x).__name__} {ids[:3]}")
    return ids


class Progress:
    """In tiến độ theo chu kỳ, kèm tốc độ và ước lượng thời gian còn lại.

    prepare_dataset với 80k dòng chạy hàng chục phút mà trước đây KHÔNG in gì cho tới khi xong
    -- nhìn y như treo. Chỉ rank 0 in, các rank khác im lặng cho đỡ rối log.
    """

    def __init__(self, label: str, total: int, every: int) -> None:
        self.label, self.total, self.every = label, total, max(1, every)
        self.t0 = time.time()
        self.quiet = os.environ.get("RANK", "0") != "0"
        self.last = 0

    def update(self, done: int) -> None:
        if self.quiet or (done < self.total and done - self.last < self.every):
            return
        self.last = done
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = done / elapsed
        eta = (self.total - done) / rate if rate > 0 else 0.0
        print(f"[prepare] {self.label}: {done:,}/{self.total:,} ({done / max(self.total, 1):.0%}) | "
              f"{rate:,.0f} dòng/s | đã {elapsed / 60:.1f} phút, còn ~{eta / 60:.1f} phút", flush=True)


def encode_record(ex: Dict[str, Any], tokenizer, args: argparse.Namespace,
                  system_prompt_override: Optional[str]) -> Dict[str, Any]:
    user_prompt = str(ex.get(args.prompt_field, "")).strip()
    if not user_prompt:
        raise ValueError(f"empty {args.prompt_field}")

    cot = str(ex.get(args.cot_field, "") or "").strip()
    if args.mode == "cot" and not cot:
        raise ValueError(f"CoT mode requires non-empty field '{args.cot_field}'")

    system_prompt = resolve_system_prompt(ex, args, system_prompt_override)

    target_json = target_to_minified_json(
        ex.get(args.target_field),
        validate=not args.skip_json_validation,
        allow_bbox_palette=args.allow_bbox_palette,
    )

    messages = build_messages(system_prompt, user_prompt, target_json, args.mode, cot)

    # Qwen3.6 does not use /think or /nothink. Its native chat template controls
    # thinking with enable_thinking. In direct mode the generation prefix contains
    # an empty <think>...</think> block; in CoT mode it ends after '<think>\n'.
    prefix = _as_ids(tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=(args.mode == "cot"),
    ))

    # preserve_thinking=True makes the full supervised sample render the assistant's
    # reasoning_content as canonical <think>...</think>, or an empty think block in no-CoT.
    full = _as_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
    ))

    # Strong sanity check: the generation prefix must exactly match the start of the
    # fully rendered supervised sample. This prevents subtle label-mask corruption.
    if full[:len(prefix)] != prefix:
        raise RuntimeError(
            "Qwen3.6 chat-template prefix mismatch. Do not train: labels would be misaligned. "
            "Check transformers/tokenizer versions."
        )

    labels = [-100] * len(prefix) + full[len(prefix):]
    keep = len(full) <= args.max_seq_length
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": labels,
        "_keep": keep,
        "_length": len(full),
    }


def prepare_dataset(path: str, tokenizer, args: argparse.Namespace,
                    system_prompt_override: Optional[str]) -> Dataset:
    """Đọc JSONL -> ids + nhãn, encode từng dòng, có in tiến độ."""
    with dbg_step(f"đọc {path}"):
        raw = read_jsonl_rows(path)  # KHÔNG dùng load_dataset("json") -- xem data_utils.py

    print(f"[prepare] {path}: {len(raw):,} dòng, bắt đầu encode", flush=True)
    progress = Progress("encode", len(raw), args.prepare_log_every)

    rows: List[Dict[str, Any]] = []
    errors = 0
    too_long = 0

    for i, ex in enumerate(raw):
        try:
            item = encode_record(ex, tokenizer, args, system_prompt_override)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"[data error] row={i}: {e}", flush=True)
            progress.update(i + 1)
            continue
        if not item.pop("_keep"):
            too_long += 1
            progress.update(i + 1)
            continue
        length = item.pop("_length")
        if DEBUG and len(rows) < 3:
            n_sup = sum(1 for lab in item["labels"] if lab != -100)
            dbg(f"row {i}: {length} token, {n_sup} token có nhãn")
        rows.append(item)
        progress.update(i + 1)

    if not rows:
        raise RuntimeError("No valid training rows remain after validation/length filtering")

    print(
        f"Prepared {len(rows):,}/{len(raw):,} rows from {path}; "
        f"invalid={errors:,}, over_max_seq={too_long:,}, max_seq={args.max_seq_length}",
        flush=True,
    )
    return Dataset.from_list(rows)


@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def cuda_mem() -> str:
    """VRAM của GPU hiện tại -- chỉ dùng cho log debug."""
    if not torch.cuda.is_available():
        return "no cuda"
    gib = 1024 ** 3
    return (f"VRAM alloc={torch.cuda.memory_allocated() / gib:.1f}G "
            f"reserved={torch.cuda.memory_reserved() / gib:.1f}G "
            f"peak={torch.cuda.max_memory_allocated() / gib:.1f}G")


def supervised_suffix_len(labels: torch.Tensor) -> int:
    """Số vị trí cuối chuỗi cần logits để phủ MỌI token có nhãn trong batch (+1 vì dịch nhãn).

    Nhãn chỉ nằm ở đuôi (sau prefix system+user), padding bên phải -> chỉ cần logits cho đoạn
    từ token có nhãn đầu tiên sớm nhất tới hết chuỗi, không phải cả ~2k token system prompt.
    """
    length = labels.shape[1]
    supervised = labels != -100
    if not bool(supervised.any()):
        return length
    first = torch.where(supervised.any(dim=1), supervised.int().argmax(dim=1),
                        torch.full_like(labels[:, 0], length))
    return min(length, length - int(first.min().item()) + 1)


class DebugStepCallback(TrainerCallback):
    """Chỉ gắn khi --debug: log mỗi optimizer step (thời gian + VRAM) để thấy chậm/treo ở step nào."""

    def __init__(self) -> None:
        self._t = time.time()

    def on_train_begin(self, args, state, control, **kw):
        dbg(f"on_train_begin: max_steps={state.max_steps} | {cuda_mem()}")
        self._t = time.time()

    def on_step_begin(self, args, state, control, **kw):
        dbg(f"step {state.global_step + 1} bắt đầu")
        self._t = time.time()

    def on_step_end(self, args, state, control, **kw):
        dbg(f"step {state.global_step} xong ({time.time() - self._t:.1f}s) | {cuda_mem()}")

    def on_evaluate(self, args, state, control, **kw):
        dbg(f"evaluate xong tại step {state.global_step}")

    def on_save(self, args, state, control, **kw):
        dbg(f"save checkpoint tại step {state.global_step}")


class BestAdapterCallback(TrainerCallback):
    """Sau MỖI lần evaluate: eval_loss thấp hơn mức tốt nhất đang có -> ghi đè <output_dir>/best_adapter.

    Không dùng load_best_model_at_end của Trainer: cái đó nạp lại bản best vào model khi kết thúc,
    làm mất bản LAST, và bản best có thể bị save_total_limit xoá mất. Ở đây best_adapter là thư mục
    riêng, final_adapter vẫn là bản last, checkpoint-* theo --save_steps chạy độc lập.
    """

    def __init__(self, best_dir: Path, write_extras, resume: bool) -> None:
        self.best_dir = best_dir
        self.write_extras = write_extras  # (dir, info) -> ghi tokenizer + system prompt + info
        self.best_loss: Optional[float] = None
        self.best_step: Optional[int] = None
        info_file = best_dir / "adapter_info.json"
        if resume and info_file.is_file():  # train tiếp: giữ mốc best của lần chạy trước
            info = json.loads(info_file.read_text(encoding="utf-8"))
            self.best_loss, self.best_step = info.get("eval_loss"), info.get("step")

    def on_train_begin(self, args, state, control, **kw):
        if state.is_world_process_zero and args.eval_strategy != "no" and args.eval_steps \
                and state.max_steps and args.eval_steps >= state.max_steps:
            print(f"[best_adapter] eval_steps={args.eval_steps} >= tổng {state.max_steps} step -> không có eval "
                  "giữa chừng, best chỉ được xét ở lần evaluate cuối (sẽ trùng last). Giảm --eval_steps.",
                  flush=True)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        loss = (metrics or {}).get("eval_loss")
        if loss is None or model is None:
            return
        if self.best_loss is not None and loss >= self.best_loss:
            if state.is_world_process_zero:
                print(f"[best_adapter] step {state.global_step}: eval_loss={loss:.4f} "
                      f"(best vẫn là {self.best_loss:.4f} @ step {self.best_step})", flush=True)
            return
        self.best_loss, self.best_step = float(loss), int(state.global_step)
        if not state.is_world_process_zero:  # mọi rank cập nhật mốc, chỉ rank 0 ghi file
            return
        model.save_pretrained(str(self.best_dir))
        self.write_extras(self.best_dir, {
            "kind": "best", "step": self.best_step, "epoch": state.epoch, "eval_loss": self.best_loss,
        })
        print(f"[best_adapter] step {self.best_step}: eval_loss={loss:.4f} -> lưu {self.best_dir}", flush=True)


class SuffixLossTrainer(Trainer):
    """Tính loss chỉ trên logits của đoạn đuôi có nhãn (logits_to_keep).

    Logits vocab ~248k cho cả chuỗi ~3k token là vài GB ở float32 -- đúng chỗ OOM trong
    cross_entropy. Kết quả loss giống hệt cách tính mặc định (các token bị bỏ vốn có nhãn -100).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Trả loss trung bình mỗi micro-batch -> để Trainer tự chia cho grad_accum.
        self.model_accepts_loss_kwargs = False

    _debug_calls = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        keep = supervised_suffix_len(labels)
        if DEBUG and self._debug_calls < 5:
            self._debug_calls += 1
            dbg(f"compute_loss #{self._debug_calls}: batch={tuple(labels.shape)} logits_to_keep={keep} "
                f"| {cuda_mem()}")
        outputs = model(**inputs, logits_to_keep=keep, use_cache=False)
        # logits vị trí [L-keep, L-1]; bỏ vị trí cuối, dự đoán token [L-keep+1, L-1].
        # Nếu bản transformers bỏ qua logits_to_keep (trả logits cả chuỗi) thì tự cắt đuôi.
        logits = outputs.logits[:, -keep:-1, :]
        targets = labels[:, labels.shape[1] - keep + 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def load_unsloth_model(args: argparse.Namespace):
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,              # A100 -> bf16 compute
        load_in_4bit=True,       # QLoRA: required for comfortable single-A100-40GB training
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha or args.lora_r,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
    )
    return model, processor


def discover_language_lora_targets(model) -> List[str]:
    targets: List[str] = []
    for name, _module in model.named_modules():
        lname = name.lower()
        if "language_model" not in lname:
            continue
        if "mtp" in lname:
            continue
        if any(name.endswith(suffix) for suffix in QWEN36_LORA_SUFFIXES):
            targets.append(name)
    if not targets:
        raise RuntimeError(
            "Could not discover Qwen3.6 language projection modules. "
            "Check Transformers/Qwen3.6 architecture version before training."
        )
    return sorted(set(targets))


def load_hf_model(args: argparse.Namespace, local_rank: int):
    """Open-source multi-GPU fallback: each DDP rank holds one 4-bit base model.

    This is data-parallel QLoRA, not tensor parallelism: every A100 stores a quantized
    copy of the base model, while only LoRA gradients are synchronized.
    """
    with dbg_step("import peft"):
        from peft import LoraConfig, get_peft_model
    import transformers
    with dbg_step("import bitsandbytes"):
        from transformers import AutoProcessor, BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  -- import sớm để lỗi/treo của bnb lộ ra ở đây

    with dbg_step(f"torch.cuda.set_device({local_rank})"):
        torch.cuda.set_device(local_rank)
    compute_dtype = torch.bfloat16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    with dbg_step(f"AutoProcessor.from_pretrained({args.model_name}) -- treo ở đây thường do thiếu HF_HUB_OFFLINE=1"):
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    # Qwen3_5ForConditionalGeneration: tuỳ bản transformers mà nằm trong mapping của
    # AutoModelForMultimodalLM hay AutoModelForImageTextToText -- thử lần lượt.
    model = None
    for auto_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        auto_cls = getattr(transformers, auto_name, None)
        if auto_cls is None:
            continue
        dbg(f"-> {auto_name}.from_pretrained (4-bit nf4) ... | {cuda_mem()}")
        try:
            model = auto_cls.from_pretrained(
                args.model_name,
                quantization_config=bnb,
                torch_dtype=compute_dtype,
                device_map={"": local_rank},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except ValueError as e:  # "Unrecognized configuration class ... for this kind of AutoModel"
            if local_rank == 0:
                print(f"HF backend: {auto_name} không nhận model này ({str(e)[:120]}), thử lớp khác")
            continue
        dbg(f"<- {auto_name}.from_pretrained xong | {cuda_mem()}")
        break
    if model is None:
        raise RuntimeError("Không lớp Auto nào của transformers nạp được model -- kiểm tra bản transformers")
    model.config.use_cache = False
    # KHÔNG dùng peft.prepare_model_for_kbit_training: nó nâng MỌI tham số không lượng tử hoá
    # (embedding + lm_head vocab ~248k, vision tower, norm) lên float32 -- thêm ~10GB/GPU, đủ
    # làm OOM A100 40GB. Giữ bf16, chỉ bật checkpointing + cho input embeddings nhận grad.
    with dbg_step("freeze + gradient_checkpointing_enable"):
        for param in model.parameters():
            param.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    with dbg_step("discover LoRA targets"):
        targets = discover_language_lora_targets(model)
    if local_rank == 0:
        suffix_counts: Dict[str, int] = {}
        for n in targets:
            s = n.rsplit(".", 1)[-1]
            suffix_counts[s] = suffix_counts.get(s, 0) + 1
        print(f"HF backend: attaching LoRA to {len(targets)} exact language modules: {suffix_counts}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha or args.lora_r,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        use_rslora=args.use_rslora,
    )
    with dbg_step("get_peft_model"):
        model = get_peft_model(model, lora_cfg)
    dbg(f"model sẵn sàng | {cuda_mem()}")
    return model, processor


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    world_size, local_rank, rank = distributed_info()
    dbg(f"args: {vars(args)}")
    dbg(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.backend == "unsloth" and world_size != 1:
        raise RuntimeError(
            "This script intentionally restricts the public Unsloth backend to one GPU. "
            "For multi-GPU OSS training use --backend hf with torchrun, or use your Unsloth Pro multi-GPU setup."
        )

    set_seed(args.seed)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    system_prompt_override = load_system_prompt_override(args.system_prompt_file)

    with dbg_step(f"load model (backend={args.backend})"):
        if args.backend == "unsloth":
            model, processor = load_unsloth_model(args)
        else:
            model, processor = load_hf_model(args, local_rank)

    tokenizer = get_text_tokenizer(processor)

    with dbg_step("prepare train dataset"):
        train_ds = prepare_dataset(args.train_file, tokenizer, args, system_prompt_override)
    with dbg_step("prepare eval dataset"):
        eval_ds = (prepare_dataset(args.eval_file, tokenizer, args, system_prompt_override)
                  if args.eval_file else None)

    denom = args.per_device_batch_size * world_size
    grad_accum = max(1, math.ceil(args.global_batch_size / denom))
    effective_global = denom * grad_accum
    if rank == 0:
        print(
            f"world_size={world_size}, micro_batch/GPU={args.per_device_batch_size}, "
            f"grad_accum={grad_accum}, effective_global_batch={effective_global}"
        )
        if effective_global != args.global_batch_size:
            print(f"[note] requested global batch {args.global_batch_size}, rounded to {effective_global}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        gradient_checkpointing=(args.backend == "hf"),
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.backend == "hf" else None,
        seed=args.seed,
        data_seed=args.seed,
        # Chỉ cần eval_loss (để chọn best_adapter). Không gom logits vocab ~248k của cả tập val.
        prediction_loss_only=True,
    )

    def write_extras(out_dir: Path, info: Dict[str, Any]) -> None:
        write_adapter_extras(out_dir, processor, tokenizer, args, system_prompt_override, info)

    best_dir = Path(args.output_dir) / "best_adapter"
    callbacks: List[TrainerCallback] = []
    if eval_ds is not None:
        callbacks.append(BestAdapterCallback(best_dir, write_extras, resume=bool(args.resume_from_checkpoint)))
    elif rank == 0:
        print("[best_adapter] không có --eval_file -> KHÔNG lưu best_adapter, chỉ có final_adapter (last)")
    if DEBUG:
        callbacks.append(DebugStepCallback())

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    # Unsloth tự vá phần tính loss; chỉ backend hf cần cắt logits về đoạn có nhãn.
    trainer_cls = SuffixLossTrainer if args.backend == "hf" else Trainer
    with dbg_step(f"tạo {trainer_cls.__name__} (bọc DDP, optimizer)"):
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=callbacks or None,
        )

    # Verify exactly where loss is active on one example before spending GPU-hours.
    if rank == 0:
        ex = train_ds[0]
        supervised = [tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100]
        print("First supervised target preview:")
        print(tokenizer.decode(supervised[:1000], skip_special_tokens=False))

    with dbg_step("trainer.train (step đầu gồm cả DDP sync -- treo ở đây thường do NCCL)"):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Evaluate thêm 1 lần trên trọng số CUỐI: step cuối thường không rơi đúng mốc eval_steps,
    # nên không có lần này thì bản last không bao giờ được xét làm best.
    last_eval_loss: Optional[float] = None
    if eval_ds is not None:
        with dbg_step("evaluate trọng số cuối"):
            last_eval_loss = trainer.evaluate().get("eval_loss")

    final_dir = Path(args.output_dir) / "final_adapter"
    with dbg_step(f"save_model -> {final_dir}"):
        trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        write_extras(final_dir, {
            "kind": "last", "step": trainer.state.global_step, "epoch": trainer.state.epoch,
            "eval_loss": last_eval_loss,
        })
        print(f"Saved LAST adapter to: {final_dir} (eval_loss={last_eval_loss})")
        best_info = best_dir / "adapter_info.json"
        if best_info.is_file():
            info = json.loads(best_info.read_text(encoding="utf-8"))
            print(f"Saved BEST adapter to: {best_dir} (step {info['step']}, eval_loss={info['eval_loss']:.4f})")


def write_adapter_extras(out_dir: Path, processor, tokenizer, args: argparse.Namespace,
                         system_prompt_override: Optional[str], info: Dict[str, Any]) -> None:
    """Ghi kèm adapter: tokenizer/processor, training_args, system prompt đã dùng, adapter_info.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        processor.save_pretrained(str(out_dir))
    except Exception:
        tokenizer.save_pretrained(str(out_dir))
    Path(out_dir, "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Lưu ĐÚNG những system prompt đã dùng lúc train -- lúc infer, app phải chọn
    # đúng file tương ứng với mức đang phục vụ (short/medium/long), không được
    # tự bịa hay dùng lẫn, nếu không sẽ lệch train/infer.
    if system_prompt_override is not None:
        Path(out_dir, "system_prompt_override.txt").write_text(system_prompt_override, encoding="utf-8")
    else:
        for level in LEVELS:
            Path(out_dir, f"system_prompt_{level}.txt").write_text(build_system_prompt(level), encoding="utf-8")
    Path(out_dir, "adapter_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
