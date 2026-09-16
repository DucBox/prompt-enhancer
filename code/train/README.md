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

## Nội dung system prompt (exp1)

Prompt ([`prompts.py`](prompts.py)) mô tả **đúng phân phối nhãn Y thật** — caption của ảnh thật
trong `test_5_new` — và mượn kỷ luật viết caption từ magic prompt v1 của Ideogram 4
(`third_party/ideogram4/src/ideogram4/magic_prompt_system_prompts/v1.txt`), nhưng **chỉ những
luật dữ liệu thật sự tuân theo**:

| Phần | Nội dung | Căn cứ trên dữ liệu |
|---|---|---|
| OUTPUT CONTRACT | một JSON minified, đúng thứ tự key, không bbox/color_palette/aspect_ratio/id | 573/573 dòng đúng thứ tự key |
| FIDELITY | giữ mọi ràng buộc; số mơ hồ giữ mơ hồ; chủ đề là hoạt động/cảnh thì phải thể hiện hoạt động | lỗi lệch chủ đề ở ảnh làm gốm (exp1 release) |
| VIETNAMESE TERMS | giữ tên Việt đủ dấu, nêu trong HLD, gloss tiếng Anh trong ngoặc tuỳ chọn | 154/199 HLD có thuật ngữ Việt; 88 ảnh có gloss, 76 ở HLD |
| FIELD GUIDE | HLD một câu; tag aesthetics ngắn; lighting = nguồn + tính chất; photo = góc + cỡ cảnh + lấy nét; desc một câu, danh tính → thuộc tính → vị trí | HLD 1 câu 182/199; desc p50 18 từ, 1 câu; 62% desc có vị trí |

**Không** mượn các luật sáng tác của v1 vì trái với caption ảnh thật: cấm "warm" (Y có ở ~20%
dòng), mặc định kiểu iPhone, "text everywhere" (cả tập chỉ 6 text element), một chủ thể = một
element (Y tách phụ kiện đeo/cầm ở 90 element), sàn luôn là background. Khi SFT, luật trái nhãn
chỉ là nhiễu — model học theo Y.

Prompt lặp lại ở mọi mẫu nên giữ dưới 1200 từ (test canh). Ước lượng mẫu dài nhất của
`test_5_new` ~3.1k token < `--max_seq_length 4096`; số chính xác xem `check_env.py --data_file`.

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

**1 GPU hay nhiều GPU:**

* `--backend unsloth` — **chỉ 1 GPU** (script chủ động chặn khi `WORLD_SIZE > 1`, vì Unsloth
  bản mã nguồn mở không hỗ trợ multi-GPU). Nhanh và tiết kiệm VRAM nhất; QLoRA 27B vừa một A100 40GB.
* `--backend hf` + `torchrun` — **nhiều GPU kiểu data-parallel (DDP)**: MỖI GPU giữ một bản
  model 4-bit đầy đủ, chỉ đồng bộ gradient LoRA. Tăng tốc theo số GPU nhưng **không chia nhỏ
  model** — mỗi GPU vẫn phải chứa được cả model (27B nf4 ~15GB, cộng embedding/lm_head bị
  `prepare_model_for_kbit_training` nâng lên float32 ~10GB, cộng activation). A100 80GB thoải
  mái; 40GB sát giới hạn, giảm `--max_seq_length` nếu OOM. `--global_batch_size` được giữ
  nguyên, grad accumulation tự chia theo số GPU.

`--mode` ở đây là chế độ suy luận (`cot` train kèm `<think>`, `no_cot` chỉ train
JSON trực tiếp) — khác với `mode` (short/medium/long) trong dữ liệu, tên trùng
nhau nhưng là hai khái niệm độc lập; xem `--detail_level_field` ở trên.

Kết quả trong `--output_dir`:

| Thư mục | Là gì |
|---|---|
| `final_adapter/` | **last** — trọng số ở step cuối |
| `best_adapter/` | **best** — trọng số có `eval_loss` thấp nhất (chỉ có khi truyền `--eval_file`) |
| `checkpoint-*/` | checkpoint theo `--save_steps`, giữ `--save_total_limit` bản gần nhất (để train tiếp) |

`final_adapter/` và `best_adapter/` đều chứa LoRA adapter + tokenizer + 3 file system prompt +
`training_args.json` + `adapter_info.json` (`kind`, `step`, `epoch`, `eval_loss`) — trỏ thẳng
`infer.py --adapter_dir` vào thư mục nào cũng được.

Best được xét sau **mỗi lần evaluate** (mỗi `--eval_steps`) và thêm một lần evaluate trên trọng số
cuối khi train xong. `--eval_steps` lớn hơn tổng số step thì không có eval giữa chừng -> best trùng
last (script sẽ in cảnh báo). Dữ liệu nhỏ (vài trăm dòng) nên đặt `--eval_steps` khoảng 5–10.

## Việc cần làm sau khi train xong

Bước 5 (đánh giá, xem `docs/plan.pdf`) — infer trên tập test bằng đúng adapter +
đúng system prompt theo mức, chấm theo `sub_json` (checklist) và so với baseline
model gốc chưa tinh chỉnh. Chưa có code cho bước này.
