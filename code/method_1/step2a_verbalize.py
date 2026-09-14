#!/usr/bin/env python3
"""STEP 2a — Sinh user prompt từ sub_json.  [GỌI MODEL]

Đầu vào : <out_root>/step1b_subjson/subjson.jsonl
Đầu ra  : <out_root>/step2a_prompts/
            prompts/<id>__<level>.json
            prompts.jsonl
            failures.json

Persona được bốc ngẫu nhiên có seed — nó CHỈ đổi văn phong, không đổi nội dung.
Nhờ vậy cùng một sub_json với persona khác nhau vẫn dùng chung một checklist chấm điểm.
Chân dung chính là "user phổ thông" (60%); các vai khác chỉ để đa dạng cách hành văn.

Mọi prompt sinh ra đều viết ĐÚNG CHÍNH TẢ. Việc chịu được đầu vào sai chính tả / mất dấu
thuộc về một mô-đun correction riêng đặt trước prompt enhancer, không trộn vào đây.

Ví dụ:
    python step2a_verbalize.py --out_root test_1 --dry_run
    python step2a_verbalize.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts

STEP = "STEP 2a"

# Trọng số: user phổ thông là chân dung chính, các vai còn lại chỉ để đa dạng văn phong.
VAI_WEIGHTED = [
    ("user phổ thông", 0.60),
    ("designer", 0.10),
    ("khách ngành cưới - sự kiện", 0.10),
    ("sinh viên làm đồ án", 0.10),
    ("nhân viên marketing", 0.10),
]
VAI = [v for v, _ in VAI_WEIGHTED]

GIONG = ["mô tả trung tính", "ra lệnh", "kể lể lan man"]

NGON_NGU_VN = "tiếng Việt"
NGON_NGU_EN = "tiếng Anh, nhưng GIỮ NGUYÊN các thuật ngữ văn hoá Việt bằng tiếng Việt"

# Không mô phỏng lỗi chính tả / mất dấu ở đây. Mọi prompt sinh ra đều coi như viết đúng.
# Nếu cần model chịu được đầu vào sai chính tả thì làm một mô-đun correction riêng
# đặt TRƯỚC prompt enhancer, không trộn nhiễu vào dữ liệu huấn luyện của bước này.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM viết user prompt từ sub_json")
    p.add_argument("--in_file", default=None, help="Mặc định: <out_root>/step1b_subjson/subjson.jsonl")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Cao để prompt đa dạng văn phong (khác step 1a dùng 0.0)")
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--english_rate", type=float, default=0.20,
                   help="Tỉ lệ prompt viết bằng tiếng Anh (giữ nguyên thuật ngữ Việt)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Chỉ dựng messages và in ra, KHÔNG gọi model")
    return io_utils.add_common_args(p).parse_args()


def sample_vai(rng: random.Random) -> str:
    """Bốc vai theo trọng số — 'user phổ thông' chiếm đa số."""
    roll = rng.random()
    cumulative = 0.0
    for name, weight in VAI_WEIGHTED:
        cumulative += weight
        if roll < cumulative:
            return name
    return VAI_WEIGHTED[-1][0]


def sample_persona(rng: random.Random, english_rate: float) -> Dict[str, str]:
    return {
        "vai": sample_vai(rng),
        "giọng": rng.choice(GIONG),
        "ngôn ngữ": NGON_NGU_EN if rng.random() < english_rate else NGON_NGU_VN,
    }


STYLE_FIELDS = ("photo", "art_style", "lighting", "aesthetics")


def build_spec(sub: Dict[str, Any], persona: Dict[str, str]) -> Dict[str, Any]:
    """Bản mô tả đưa cho model — chứa ĐÚNG các mệnh đề trong checklist, không hơn.

    Không đưa tên nhóm thô (thường chỉ là nhãn như "cảnh nền"): model sẽ nhắc tới nó và
    bị 2b chấm là thêm tin.
    """
    spec: Dict[str, Any] = {
        "detail_level": sub["detail_level"],
        "length_hint": sub["length_hint"],
        "persona": persona,
        "groups": [
            {"facts": g["facts"], **({"so_nhieu": True} if g.get("so_nhieu") else {})}
            for g in sub["groups"]
        ],
    }
    if sub.get("background"):
        spec["boi_canh"] = sub["background"]
    phong_cach = [f for key in STYLE_FIELDS for f in (sub.get("style") or {}).get(key, [])]
    if sub.get("medium"):
        phong_cach.append(sub["medium"])
    if phong_cach:
        spec["phong_cach"] = phong_cach
    return spec


def clean_prompt_text(text: str) -> str:
    """Gỡ ngoặc kép bao ngoài và gộp xuống dòng — model hay trả về dạng đó."""
    s = " ".join(str(text).split())
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", "“"):
        s = s[1:-1].strip()
    if s.startswith("“") and s.endswith("”"):
        s = s[1:-1].strip()
    return s


def key_of(sub: Dict[str, Any]) -> str:
    return "{}__{}".format(sub["id"], sub["detail_level"])


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "LLM viết user prompt từ sub_json")
    config.load_env(args.env_file)

    in_file = io_utils.resolve(args.in_file, args.out_root, "step1b", "subjson.jsonl")
    subs = io_utils.read_jsonl(in_file)
    subs = io_utils.apply_test_mode(subs, args, "sub_json")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step2a")
    cache_dir = out_dir / "prompts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Persona bốc theo seed + key nên chạy lại vẫn ra đúng persona cũ.
    specs: Dict[str, Dict[str, Any]] = {}
    for sub in subs:
        key = key_of(sub)
        rng = random.Random("{}|persona|{}".format(args.seed, key))
        persona = sample_persona(rng, args.english_rate)
        specs[key] = build_spec(sub, persona)

    if args.dry_run:
        sub = subs[0]
        messages = prompts.build_step2a_messages(specs[key_of(sub)])
        print("\n[DRY RUN] {} messages cho {}".format(len(messages), key_of(sub)))
        print("\n--- system ---\n{}".format(messages[0]["content"][:900]))
        print("\n--- user cuối cùng ---\n{}".format(messages[-1]["content"]))
        return

    todo = []
    n_cached = 0
    for sub in subs:
        if not args.overwrite and (cache_dir / "{}.json".format(key_of(sub))).is_file():
            n_cached += 1
            continue
        todo.append(sub)

    print("đã có cache: {} | cần gọi model: {}".format(n_cached, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo:
        client = llm.LLMClient()
        print("endpoint : {}".format(client.endpoint))
        print("model    : {}\n".format(client.model))

        def process(sub: Dict[str, Any]) -> Optional[str]:
            key = key_of(sub)
            messages = prompts.build_step2a_messages(specs[key])
            raw = client.chat(
                messages, temperature=args.temperature, max_tokens=args.max_tokens,
            )
            text = clean_prompt_text(raw)
            if not text:
                raise llm.LLMError("model trả về chuỗi rỗng")
            io_utils.write_json(cache_dir / "{}.json".format(key), {
                "id": sub["id"],
                "detail_level": sub["detail_level"],
                "persona": specs[key]["persona"],
                "user_prompt": text,
                "n_words": len(text.split()),
            })
            return key

        def on_error(sub: Dict[str, Any], exc: Exception) -> None:
            failures.append({"key": key_of(sub), "error": str(exc)[:400]})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="sinh prompt", on_error=on_error)

    merged: List[Dict[str, Any]] = []
    for sub in subs:
        path = cache_dir / "{}.json".format(key_of(sub))
        if not path.is_file():
            continue
        item = io_utils.read_json(path)
        merged.append({
            "id": sub["id"],
            "detail_level": sub["detail_level"],
            "persona": item["persona"],
            "user_prompt": item["user_prompt"],
            "n_words": item["n_words"],
            "checklist": sub["checklist"],
            "required_subject": sub.get("required_subject"),
        })

    n_written = io_utils.write_jsonl(out_dir / "prompts.jsonl", merged)
    if failures:
        io_utils.write_json(out_dir / "failures.json", failures)

    print("\n--- Độ dài prompt theo mức ---")
    for level in ("short", "medium", "long"):
        items = [m for m in merged if m["detail_level"] == level]
        if not items:
            continue
        lengths = sorted(m["n_words"] for m in items)
        avg = sum(lengths) / len(lengths)
        print("  {:<8} n={:<6} trung bình {:>6.1f} từ   p50={:<4} min={:<4} max={}".format(
            level, len(items), avg, lengths[len(lengths) // 2], lengths[0], lengths[-1]))

    io_utils.summary(**{
        "sub_json xử lý": len(subs),
        "dùng lại cache": n_cached,
        "gọi model": len(todo),
        "thất bại": len(failures),
        "ghi ra prompts.jsonl": n_written,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
