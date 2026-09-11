#!/usr/bin/env python3
"""Normalize raw Ideogram-style caption files into Prompt-Enhancer SFT targets.

Input : a directory of *.txt / *.json files, each holding one JSON object.
Output: - one .json per sample (pretty, ensure_ascii=False) in --out_dir
        - one targets.jsonl with {"id", "target_json"} rows, ready to join with
          generated user prompts later.

Normalization rules:
  * drop `bbox` and `color_palette` anywhere in the tree
  * enforce the top-level schema (high_level_description / style_description /
    compositional_deconstruction) and report violations instead of silently passing
  * enforce `photo` XOR `art_style` inside style_description
  * keep key ordering stable so the SFT target string is deterministic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DROP_KEYS = {"bbox", "color_palette"}

TOP_ORDER = ["high_level_description", "style_description", "compositional_deconstruction"]
STYLE_ORDER = ["aesthetics", "lighting", "photo", "art_style", "medium"]
COMP_ORDER = ["background", "elements"]
ELEM_ORDER = ["type", "text", "desc"]


def strip_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip_keys(x) for x in obj]
    return obj


def reorder(d: Dict[str, Any], order: List[str]) -> Dict[str, Any]:
    out = {k: d[k] for k in order if k in d}
    out.update({k: v for k, v in d.items() if k not in out})
    return out


def validate(obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for k in TOP_ORDER:
        if k not in obj:
            errs.append(f"missing top-level '{k}'")

    style = obj.get("style_description")
    if not isinstance(style, dict):
        errs.append("style_description is not an object")
    else:
        has_photo, has_art = "photo" in style, "art_style" in style
        if has_photo and has_art:
            errs.append("style_description has both photo and art_style")
        elif not has_photo and not has_art:
            errs.append("style_description has neither photo nor art_style")

    comp = obj.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        errs.append("compositional_deconstruction is not an object")
    else:
        if "background" not in comp:
            errs.append("compositional_deconstruction.background missing")
        elements = comp.get("elements")
        if not isinstance(elements, list) or not elements:
            errs.append("compositional_deconstruction.elements missing or empty")
        else:
            for i, el in enumerate(elements):
                if not isinstance(el, dict):
                    errs.append(f"elements[{i}] is not an object")
                    continue
                etype = el.get("type")
                if etype not in {"obj", "text"}:
                    errs.append(f"elements[{i}].type={etype!r} not in (obj, text)")
                if etype == "text" and not el.get("text"):
                    errs.append(f"elements[{i}] type=text without verbatim 'text'")
                if not el.get("desc"):
                    errs.append(f"elements[{i}] missing 'desc'")
    return errs


def normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    obj = strip_keys(raw)
    if isinstance(obj.get("style_description"), dict):
        obj["style_description"] = reorder(obj["style_description"], STYLE_ORDER)
    comp = obj.get("compositional_deconstruction")
    if isinstance(comp, dict):
        if isinstance(comp.get("elements"), list):
            comp["elements"] = [
                reorder(el, ELEM_ORDER) if isinstance(el, dict) else el
                for el in comp["elements"]
            ]
        obj["compositional_deconstruction"] = reorder(comp, COMP_ORDER)
    return reorder(obj, TOP_ORDER)


def process(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("file does not contain a JSON object")
    obj = normalize(raw)
    return obj, validate(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--jsonl", default=None, help="path of the combined targets.jsonl")
    ap.add_argument("--strict", action="store_true", help="skip samples that fail validation")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "targets.jsonl"

    files = sorted(p for p in in_dir.iterdir() if p.suffix in {".txt", ".json"})
    ok = bad = skipped = 0
    rows: List[str] = []

    for p in files:
        try:
            obj, errs = process(p)
        except Exception as e:  # unparseable file
            bad += 1
            print(f"[parse error] {p.name}: {e}")
            continue
        if errs:
            bad += 1
            print(f"[schema] {p.name}: " + "; ".join(errs))
            if args.strict:
                skipped += 1
                continue
        (out_dir / f"{p.stem}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(json.dumps({"id": p.stem, "target_json": obj}, ensure_ascii=False))
        ok += 1

    jsonl_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {ok}/{len(files)} samples -> {out_dir}")
    print(f"jsonl -> {jsonl_path}")
    print(f"files with schema warnings: {bad}, skipped: {skipped}")


if __name__ == "__main__":
    main()
