# Train — SFT Prompt Enhancer (Qwen3.6-27B, LoRA)

Bước 4 của plan: nhận `train.jsonl`/`val.jsonl` từ
[`code/method_1/step2c_split.py`](../method_1/step2c_split.py), train MỘT model
(LoRA adapter) làm nhiệm vụ `user_prompt → target_json` cho cả 3 mức short/medium/long.

Chạy trên server có GPU — file này import `torch`/`transformers`/`datasets` nên
**không chạy/test được trên máy không có các thư viện đó**. Phần logic thuần (ghép
system prompt theo mức) tách riêng ở [`prompts.py`](prompts.py), test được ở
[`tests/test_prompts.py`](tests/test_prompts.py) mà không cần GPU:

```bash
python3 -m unittest discover -s tests -p "test_prompts.py"
```

## Quyết định thiết kế: mỗi mức một system prompt riêng

Train chung MỘT model cho cả 3 mức, nhưng **mỗi mức dùng một system prompt khác
nhau** (không phải một prompt chung kèm nhãn `DETAIL_LEVEL: short`) — vì việc model
cần làm ở mỗi mức thực sự khác nhau:

* **short** — input chỉ 8-20 từ, nêu vài ý chính. Model phải **tự bổ sung** chi
  tiết hợp lý (ánh sáng, bối cảnh, chất liệu...) để JSON vẫn đầy đủ như một caption
  chuyên nghiệp — không được để JSON nghèo nàn theo độ ngắn của input.
* **medium** — input 30-60 từ, đã nêu vài ý rõ. Giữ nguyên phần đã có, lấp phần
  còn thiếu.
* **long** — input 100-200 từ, gần như đầy đủ. Việc chính là **cấu trúc hoá trung
  thành** những gì user đã mô tả, hạn chế tối đa tự thêm.

Hệ quả bắt buộc: **ứng dụng gọi model lúc inference phải biết trước đang phục vụ
mức nào** (short/medium/long) để chọn đúng system prompt — giống hệt lúc train.
Không có cách nào để model tự đoán ra mức mà không lệch với lúc train. Ba file
system prompt được lưu lại trong adapter output (`system_prompt_short.txt`,
`system_prompt_medium.txt`, `system_prompt_long.txt`) — app tầng trên đọc đúng
file tương ứng khi gọi model.

## Chạy train

`step2c_split.py` sinh sẵn field `"mode"` (short/medium/long) trong mỗi dòng —
script train đọc field này để tự chọn đúng system prompt (`--detail_level_field`,
mặc định `mode`, không cần chỉnh nếu dùng đúng pipeline ở `code/method_1`).

```bash
# 1 GPU (khuyến nghị, Unsloth QLoRA)
python3 train_prompt_enhancer_qwen36.py \
  --train_file  ../method_1/outputs/run_full/step2c_split/train.jsonl \
  --eval_file   ../method_1/outputs/run_full/step2c_split/val.jsonl \
  --output_dir  ./runs/pe_v1 \
  --mode no_cot \
  --backend unsloth

# nhiều GPU (fallback HF/PEFT + DDP)
torchrun --nproc_per_node=4 train_prompt_enhancer_qwen36.py \
  --train_file  ../method_1/outputs/run_full/step2c_split/train.jsonl \
  --eval_file   ../method_1/outputs/run_full/step2c_split/val.jsonl \
  --output_dir  ./runs/pe_v1 \
  --mode no_cot \
  --backend hf
```

`--mode` ở đây là chế độ suy luận (`cot` train kèm `<think>`, `no_cot` chỉ train
JSON trực tiếp) — khác với `mode` (short/medium/long) trong dữ liệu, tên trùng
nhau nhưng là hai khái niệm độc lập; xem `--detail_level_field` ở trên.

Kết quả: `./runs/pe_v1/final_adapter/` chứa LoRA adapter + tokenizer + 3 file
system prompt + `training_args.json` (để tái lập chính xác lần train này).

## Việc cần làm sau khi train xong

Bước 5 (đánh giá, xem `docs/plan.pdf`) — infer trên tập test bằng đúng adapter +
đúng system prompt theo mức, chấm theo `sub_json` (checklist) và so với baseline
model gốc chưa tinh chỉnh. Chưa có code cho bước này.
