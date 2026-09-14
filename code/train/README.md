# Train — SFT Prompt Enhancer (Qwen3.6-27B, LoRA)

Bước 4 của plan: nhận `train.jsonl`/`val.jsonl` từ `step2d_final/`
([`code/method_1/step2d_finalize.py`](../method_1/step2d_finalize.py)), train MỘT model
(LoRA adapter) làm nhiệm vụ `user_prompt → target_json` cho cả 3 mức short/medium/long.

Chạy trên server có GPU — file này import `torch`/`transformers`/`datasets` nên
**không chạy/test được trên máy không có các thư viện đó**. Phần logic thuần tách riêng,
test được mà không cần GPU:

* [`prompts.py`](prompts.py) — ghép system prompt theo mức.
* [`data_utils.py`](data_utils.py) — đọc JSONL và kiểm tra target.

```bash
python3 -m unittest discover -s tests
```

**Không đọc dữ liệu bằng `datasets.load_dataset("json")`.** Arrow gộp `target_json` của mọi
dòng thành một schema chung và chèn key thiếu với giá trị `null` (`"art_style": null` vào
target ảnh chụp, `"photo": null` vào target tranh/3D, `"text": null` vào mọi element obj) —
model sẽ học sinh ra các key đó. Script đọc từng dòng bằng `json.loads` và từ chối target
còn `null`, `id`, `bbox` hoặc `color_palette`.

## Quyết định thiết kế: mỗi mức một system prompt riêng

Train chung MỘT model cho cả 3 mức, nhưng **mỗi mức dùng một system prompt khác
nhau** (không phải một prompt chung kèm nhãn `DETAIL_LEVEL: short`) — vì việc model
cần làm ở mỗi mức thực sự khác nhau:

* **short** — user chỉ nêu chủ thể chính cùng 1–2 ý (thuộc tính nổi bật, số lượng hoặc
  một địa điểm có bản sắc). Model phải **tự bổ sung** chi tiết hợp lý (ánh sáng, bối
  cảnh, chất liệu...) để JSON vẫn đầy đủ như một caption chuyên nghiệp — không được để
  JSON nghèo nàn theo độ ngắn của input.
* **medium** — user tả các phần chính (chủ thể với vài thuộc tính, 1–3 chủ thể phụ, bối
  cảnh chính). Giữ nguyên phần đã có, lấp phần còn thiếu.
* **long** — user tả gần như toàn bộ ảnh. Việc chính là **cấu trúc hoá trung thành**
  những gì user đã mô tả, hạn chế tối đa tự thêm.

Mức được định nghĩa theo NỘI DUNG (giống step 1b), không theo số từ: trên `test_5_new`
prompt long dài 54–402 từ, nên một khoảng số từ cố định trong system prompt sẽ mô tả sai.

Hệ quả bắt buộc: **ứng dụng gọi model lúc inference phải biết trước đang phục vụ
mức nào** (short/medium/long) để chọn đúng system prompt — giống hệt lúc train.
Không có cách nào để model tự đoán ra mức mà không lệch với lúc train. Ba file
system prompt được lưu lại trong adapter output (`system_prompt_short.txt`,
`system_prompt_medium.txt`, `system_prompt_long.txt`) — app tầng trên đọc đúng
file tương ứng khi gọi model.

## Kiểm tra môi trường trước khi train

```bash
python3 check_env.py                    # backend unsloth; thêm --backend hf nếu dùng torchrun
python3 check_env.py --data_file ../method_1/outputs/test_5_new/step2d_final/train.jsonl
```

Script kiểm tra Python ≥ 3.10, các gói bắt buộc, `transformers` có kiến trúc `qwen3_5`
chưa, CUDA / VRAM / bf16, import được script train. Với `--data_file`, nó tải tokenizer
(không tải trọng số) và encode thử vài dòng bằng đúng hàm của script train. Cuối cùng in
lệnh `pip install` cho những gì còn thiếu; thoát mã 1 nếu chưa đủ để train.

## Chạy train

Dùng `step2d_final/` — **không** dùng `step2c_split/` (target ở đó còn `id`; script sẽ từ
chối). Mỗi dòng có field `"mode"` (short/medium/long) — script train đọc field này để tự
chọn đúng system prompt (`--detail_level_field`, mặc định `mode`). Dữ liệu hiện chưa có
CoT nên dùng `--mode no_cot`.

```bash
# 1 GPU (khuyến nghị, Unsloth QLoRA)
python3 train_prompt_enhancer_qwen36.py \
  --train_file  ../method_1/outputs/run_full/step2d_final/train.jsonl \
  --eval_file   ../method_1/outputs/run_full/step2d_final/val.jsonl \
  --output_dir  ./runs/pe_v1 \
  --mode no_cot \
  --backend unsloth

# nhiều GPU (fallback HF/PEFT + DDP)
torchrun --nproc_per_node=4 train_prompt_enhancer_qwen36.py \
  --train_file  ../method_1/outputs/run_full/step2d_final/train.jsonl \
  --eval_file   ../method_1/outputs/run_full/step2d_final/val.jsonl \
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
