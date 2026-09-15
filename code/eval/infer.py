#!/usr/bin/env python3
"""Infer tập test bằng model gốc (baseline) và/hoặc model gốc + LoRA adapter -> predictions.

Chọn biến thể bằng 2 cờ, bật cờ nào chạy cờ đó (bật cả 2 thì chạy cả 2, chỉ load model 1 lần):
    --run_baseline   model gốc, KHÔNG adapter
    --run_adapter    model gốc + adapter (--adapter_dir)

Dựng prompt GIỐNG HỆT lúc train (code/train/train_prompt_enhancer_qwen36.py, --mode no_cot):
    system = system prompt theo mức của dòng (short/medium/long)
    user   = user_prompt
    chat template Qwen3.6 với enable_thinking=False -> prefix có khối <think></think> rỗng,
    model sinh thẳng JSON.
System prompt lấy từ <adapter_dir>/system_prompt_<mức>.txt (đúng bản đã dùng lúc train); không có
adapter_dir thì lấy từ code/train/prompts.py. Baseline dùng CÙNG system prompt với adapter để so
công bằng.

Đầu ra (<output_dir>):
    raw/<biến thể>/<id>__<mức>.json   cache từng dòng (chạy lại sẽ bỏ qua dòng đã có)
    predictions_<biến thể>.jsonl      gộp theo đúng thứ tự tập test -- đầu vào cho bước chấm điểm
    run_config.json                   tham số + nguồn system prompt

Nhiều GPU: chạy bằng torchrun, mỗi rank giữ 1 bản model và xử lý 1 phần dữ liệu (chia theo dòng).

Ví dụ:
    python -m torch.distributed.run --nproc_per_node=8 infer.py \\
        --model_name /models/Qwen3.6-27B --adapter_dir runs/exp4/final_adapter \\
        --test_file data/step2d_final/test.jsonl --output_dir eval_runs/exp4 \\
        --run_baseline --run_adapter
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))

from data_utils import read_jsonl_rows  # noqa: E402
from prompts import LEVELS, build_system_prompt  # noqa: E402

BASELINE = "baseline"
ADAPTER = "adapter"


def _argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


# --model_name là folder local -> bật offline HF TRƯỚC khi import transformers (server không ra mạng).
if __name__ == "__main__":
    _model_arg = _argv_value(sys.argv[1:], "--model_name")
    if _model_arg and os.path.isdir(_model_arg):
        for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.setdefault(_var, "1")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer baseline và/hoặc adapter trên tập test")
    p.add_argument("--model_name", required=True, help="Folder model gốc (vd Qwen3.6-27B)")
    p.add_argument("--adapter_dir", default=None, help="Folder final_adapter (bắt buộc khi --run_adapter)")
    p.add_argument("--test_file", required=True, help="step2d_final/test.jsonl")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--run_baseline", action="store_true", help="Chạy model gốc, không adapter")
    p.add_argument("--run_adapter", action="store_true", help="Chạy model gốc + adapter")

    p.add_argument("--modes", nargs="+", default=list(LEVELS), choices=list(LEVELS),
                   help="Chỉ chạy các mức này (mặc định cả 3)")
    p.add_argument("--limit", type=int, default=None, help="Chỉ lấy N dòng đầu (thử nhanh)")

    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Số dòng sinh cùng lúc mỗi GPU (padding trái). 1 là an toàn nhất.")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                   help="Base 4-bit nf4 như lúc train (mặc định). --no-load_in_4bit = bf16.")
    p.add_argument("--overwrite", action="store_true", help="Sinh lại cả dòng đã có cache")
    p.add_argument("--dist_timeout_hours", type=float, default=12.0,
                   help="Thời gian các rank chờ nhau ở cuối trước khi gộp file")

    args = p.parse_args(argv)
    if not (args.run_baseline or args.run_adapter):
        p.error("cần bật ít nhất một cờ: --run_baseline và/hoặc --run_adapter")
    if args.run_adapter and not args.adapter_dir:
        p.error("--run_adapter cần --adapter_dir")
    if args.adapter_dir and not Path(args.adapter_dir).is_dir():
        p.error(f"--adapter_dir không tồn tại: {args.adapter_dir}")
    return args


def selected_variants(args: argparse.Namespace) -> List[str]:
    """Adapter chạy trước: đó là kết quả cần xem nhất, lỡ dừng giữa chừng vẫn có."""
    return [v for v, on in ((ADAPTER, args.run_adapter), (BASELINE, args.run_baseline)) if on]


def load_system_prompts(adapter_dir: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Trả (prompt theo mức, nguồn theo mức). Ưu tiên file đã lưu cùng adapter lúc train."""
    prompts: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for level in LEVELS:
        path = Path(adapter_dir) / f"system_prompt_{level}.txt" if adapter_dir else None
        if path is not None and path.is_file():
            prompts[level] = path.read_text(encoding="utf-8")
            sources[level] = str(path)
        else:
            prompts[level] = build_system_prompt(level)
            sources[level] = "code/train/prompts.py"
    return prompts, sources


def prompt_drift(prompts: Dict[str, str]) -> List[str]:
    """Các mức mà system prompt của adapter khác prompts.py hiện tại (code đã sửa sau khi train)."""
    return [level for level in LEVELS if prompts[level].strip() != build_system_prompt(level).strip()]


def row_key(row: Dict[str, Any]) -> str:
    return "{}__{}".format(row["id"], row["mode"])


def select_rows(rows: List[Dict[str, Any]], modes: Sequence[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    picked = [r for r in rows if r.get("mode") in modes]
    return picked[:limit] if limit else picked


def shard(rows: List[Any], rank: int, world_size: int) -> List[Any]:
    return rows[rank::world_size]


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.I)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Lấy object JSON từ đầu ra model. Trả (object, None) hoặc (None, lý do lỗi).

    Chịu được: khối <think>...</think> sót lại, rào ```json, chữ thừa trước/sau object.
    """
    s = text
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", s.strip()))
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            return None, "không tìm thấy object JSON"
        try:
            obj = json.loads(s[start:end + 1])
        except json.JSONDecodeError as e:
            return None, f"JSON lỗi: {e.msg} (vị trí {e.pos})"
    if not isinstance(obj, dict):
        return None, "JSON không phải object"
    return obj, None


def build_record(row: Dict[str, Any], variant: str, raw_output: str, n_input_tokens: int,
                 n_output_tokens: int, gen_seconds: float, max_new_tokens: int) -> Dict[str, Any]:
    pred, error = extract_json(raw_output)
    return {
        "id": row["id"],
        "mode": row["mode"],
        "variant": variant,
        "user_prompt": row["user_prompt"],
        "raw_output": raw_output,
        "pred_json": pred,
        "parse_error": error,
        "n_input_tokens": n_input_tokens,
        "n_output_tokens": n_output_tokens,
        "hit_max_new_tokens": n_output_tokens >= max_new_tokens,
        "gen_seconds": round(gen_seconds, 2),
        # Mang theo để bước chấm điểm không phải đọc lại tập test.
        "target_json": row.get("target_json"),
        "sub_json": row.get("sub_json"),
    }


def cache_path(output_dir: str, variant: str, row: Dict[str, Any]) -> Path:
    return Path(output_dir) / "raw" / variant / f"{row_key(row)}.json"


def merge_predictions(output_dir: str, variant: str, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Gộp cache thành predictions_<biến thể>.jsonl theo đúng thứ tự tập test."""
    out_file = Path(output_dir) / f"predictions_{variant}.jsonl"
    written = missing = parse_failed = truncated = 0
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            path = cache_path(output_dir, variant, row)
            if not path.is_file():
                missing += 1
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            parse_failed += record["pred_json"] is None
            truncated += bool(record["hit_max_new_tokens"])
    return {"written": written, "missing": missing, "parse_failed": parse_failed, "truncated": truncated}


def log(rank: int, msg: str) -> None:
    print(f"[infer r{rank}] {msg}", flush=True)


# --------------------------------------------------------------------------------------- GPU


def load_model(args: argparse.Namespace, local_rank: int, world_size: int, with_adapter: bool):
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"  # sinh theo batch: prompt phải thẳng hàng bên phải
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # Nhiều rank: mỗi rank 1 GPU. Một process bf16: trải model qua các GPU nhìn thấy.
    kwargs["device_map"] = {"": local_rank} if (world_size > 1 or args.load_in_4bit) else "auto"
    if world_size > 1 or args.load_in_4bit:
        torch.cuda.set_device(local_rank)

    model = None
    for auto_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        auto_cls = getattr(transformers, auto_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(args.model_name, **kwargs)
        except ValueError:
            continue
        break
    if model is None:
        raise RuntimeError("Không lớp Auto nào của transformers nạp được model")

    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, tokenizer


def stop_token_ids(tokenizer) -> List[int]:
    ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


def generate_batch(model, tokenizer, texts: List[str], max_new_tokens: int) -> List[Tuple[str, int, int]]:
    """Sinh greedy cho một batch. Trả [(chuỗi đầu ra, số token vào, số token ra)]."""
    import torch

    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            eos_token_id=stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    prompt_len = enc["input_ids"].shape[1]
    stops = set(stop_token_ids(tokenizer)) | {tokenizer.pad_token_id}
    results = []
    for i in range(len(texts)):
        new = out[i, prompt_len:].tolist()
        n_out = next((j + 1 for j, t in enumerate(new) if t in stops), len(new))
        text = tokenizer.decode(new[:n_out], skip_special_tokens=True)
        results.append((text, int(enc["attention_mask"][i].sum().item()), n_out))
    return results


def run_variant(model, tokenizer, variant: str, rows: List[Dict[str, Any]], prompts: Dict[str, str],
                args: argparse.Namespace, rank: int) -> None:
    import contextlib

    todo = [r for r in rows if args.overwrite or not cache_path(args.output_dir, variant, r).is_file()]
    log(rank, f"{variant}: {len(rows)} dòng của rank này, đã có cache {len(rows) - len(todo)}, cần sinh {len(todo)}")
    if not todo:
        return
    (Path(args.output_dir) / "raw" / variant).mkdir(parents=True, exist_ok=True)

    # Cả 2 biến thể dùng chung 1 model: baseline = tắt adapter tạm thời.
    ctx = model.disable_adapter() if (variant == BASELINE and hasattr(model, "disable_adapter")) \
        else contextlib.nullcontext()
    done = 0
    with ctx:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    build_messages(prompts[r["mode"]], r["user_prompt"]),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for r in batch
            ]
            t0 = time.time()
            outputs = generate_batch(model, tokenizer, texts, args.max_new_tokens)
            seconds = (time.time() - t0) / len(batch)
            for row, (text, n_in, n_out) in zip(batch, outputs):
                record = build_record(row, variant, text, n_in, n_out, seconds, args.max_new_tokens)
                path = cache_path(args.output_dir, variant, row)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                done += 1
                status = "ok" if record["pred_json"] is not None else f"LỖI JSON ({record['parse_error']})"
                if record["hit_max_new_tokens"]:
                    status += " | CHẠM max_new_tokens"
                log(rank, f"{variant} {done}/{len(todo)} {row_key(row)}: vào {n_in} tok, ra {n_out} tok, "
                          f"{seconds:.1f}s | {status}")


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    variants = selected_variants(args)
    all_rows = select_rows(read_jsonl_rows(args.test_file), args.modes, args.limit)
    if not all_rows:
        raise SystemExit(f"Không có dòng nào sau khi lọc mức {args.modes} từ {args.test_file}")
    prompts, sources = load_system_prompts(args.adapter_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0:
        log(rank, f"biến thể: {variants} | {len(all_rows)} dòng | world_size={world_size} | "
                  f"4bit={args.load_in_4bit} | batch={args.batch_size} | max_new_tokens={args.max_new_tokens}")
        for level in LEVELS:
            log(rank, f"system prompt {level}: {sources[level]}")
        drift = prompt_drift(prompts)
        if args.adapter_dir and drift:
            log(rank, f"[chú ý] system prompt của adapter KHÁC prompts.py hiện tại ở mức {drift} "
                      "-- vẫn dùng bản của adapter (đúng bản lúc train)")
        Path(args.output_dir, "run_config.json").write_text(json.dumps({
            "args": vars(args), "variants": variants, "n_rows": len(all_rows),
            "world_size": world_size, "system_prompt_sources": sources,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if world_size > 1:
        import datetime
        import torch.distributed as dist
        # gloo chỉ để các rank chờ nhau ở cuối (barrier) -- không truyền tensor nào qua NCCL.
        dist.init_process_group("gloo", timeout=datetime.timedelta(hours=args.dist_timeout_hours))

    my_rows = shard(all_rows, rank, world_size)
    t0 = time.time()
    log(rank, f"load model{' + adapter' if args.run_adapter else ''} ...")
    model, tokenizer = load_model(args, local_rank, world_size, with_adapter=args.run_adapter)
    log(rank, f"load xong ({time.time() - t0:.0f}s)")

    for variant in variants:
        run_variant(model, tokenizer, variant, my_rows, prompts, args, rank)
    log(rank, "xong phần của rank này")

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if rank == 0:
        for variant in variants:
            stats = merge_predictions(args.output_dir, variant, all_rows)
            log(rank, f"{variant}: ghi {stats['written']} dòng -> predictions_{variant}.jsonl | "
                      f"thiếu {stats['missing']} | lỗi JSON {stats['parse_failed']} | "
                      f"chạm max_new_tokens {stats['truncated']}")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
