# Method 1 — Sinh dữ liệu huấn luyện cho Prompt Enhancer

Pipeline đi ngược từ `target_json` (caption của ảnh thật) ra `user_prompt`,
rồi train theo chiều xuôi `user_prompt → target_json`.

Kế hoạch đầy đủ kèm ví dụ: [`docs/plan.pdf`](../../docs/plan.pdf).

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env      # rồi điền endpoint + model thật
```

`.env` và mọi thư mục `output/`, `data/` đều đã nằm trong `.gitignore`.
Endpoint không bao giờ xuất hiện trong source — có test tự động canh việc này
(`test_no_endpoint_hardcoded_in_source`).

## Các bước

| Step | File | Gọi model | Thư mục con trong `--out_dir` |
|------|------|:---------:|--------|
| 0  | `step0_normalize.py`       | –   | `step0_normalized/` |
| 1a | `step1a_decompose.py`      | ✅  | `step1a_decompose/` |
| 1b | `step1b_build_subjson.py`  | ✅  | `step1b_subjson/` |
| 2a | `step2a_verbalize.py`      | ✅  | `step2a_prompts/` |
| 2b | `step2b_filter.py`         | ✅  | `step2b_filtered/` |
| 2b2 | `step2b2_correct.py` (tuỳ chọn, `--retry_rejected`) | ✅  | `step2b2_corrected/` |
| 2c | `step2c_split.py`          | –   | `step2c_split/` |
| 2d | `step2d_finalize.py`       | –   | `step2d_final/` |

Mỗi step ghi vào một thư mục con riêng, và tự tìm đầu ra của step trước trong cùng
thư mục gốc đó.

## Chạy

```bash
# chạy thử 20 mẫu, kết quả vào test_1/
bash run_all.sh --in_dir /duong/dan/data --out_dir test_1 --test

# chạy full
bash run_all.sh --in_dir /duong/dan/data --out_dir run_full

# chạy lại từ giữa chừng (không cần --in_dir)
bash run_all.sh --out_dir test_1 --from 2a
```

Cây đầu ra:

```
test_1/
├── step0_normalized/     targets.jsonl, targets/, audit_report.json
├── step1a_decompose/     decompose.jsonl, decompose/<id>.json
├── step1b_subjson/       subjson.jsonl, selection/<id>.json, stats.json
├── step2a_prompts/       prompts.jsonl, prompts/<id>__<level>.json
├── step2b_filtered/      passed.jsonl, rejected.jsonl, report.json
├── step2b2_corrected/    corrected_passed.jsonl, rejected_final.jsonl (chỉ có nếu --retry_rejected)
├── step2c_split/         train.jsonl, val.jsonl, test.jsonl
└── step2d_final/         train.jsonl, val.jsonl, test.jsonl  <- DỮ LIỆU HUẤN LUYỆN
```

| Cờ của `run_all.sh` | Mặc định | |
|---|---|---|
| `--in_dir DIR` | – | thư mục data gốc (bắt buộc trừ khi dùng `--from`) |
| `--out_dir DIR` | `output` | thư mục output gốc |
| `--test` | tắt | chạy thử trên ít mẫu |
| `--test_samples N` | 20 | |
| `--workers N` | 4 | số luồng gọi model |
| `--from STEP` | `0` | bắt đầu từ `0\|1a\|1b\|2a\|2b\|2b2\|2c\|2d` |
| `--retry_rejected` | tắt | bật step 2b2: sửa lại (retry ĐÚNG 1 lần) prompt bị 2b loại |
| `--ignore_judge` | tắt | tắt hẳn cổng 2b — không gọi judge, coi mọi prompt là đạt |

**`--test` chỉ được đặt ở step 0.** Step 0 lọc dữ liệu còn N mẫu, các step sau tự kế
thừa N mẫu đó. Nếu đặt `--test` ở mọi step thì mỗi step lại bốc ngẫu nhiên tiếp và
chuỗi dữ liệu bị đứt — `run_all.sh` đã xử lý đúng việc này.

Chạy từng step riêng cũng được, chỉ cần truyền `--out_root`:

```bash
python3 step0_normalize.py --in_dir DATA --out_root test_1 --test --test_samples 20
python3 step1a_decompose.py --out_root test_1 --workers 4
python3 step1b_build_subjson.py --out_root test_1
```

Ba step gọi model có `--dry_run`: dựng messages và in ra màn hình mà **không** gọi
mạng — dùng để soi prompt trước khi đốt tiền.

### Ghi chú vận hành

- **Cache & resume.** Step 1a / 1b / 2a / 2b lưu kết quả từng mẫu ra file riêng. Chạy lại
  sẽ bỏ qua mẫu đã xong, nên có thể dừng giữa chừng rồi chạy tiếp. Dùng `--overwrite`
  để gọi lại từ đầu.
- **Song song.** `--workers N` (mặc định 4).
- **1a phân rã, 1b chọn lọc — cả hai là LLM, không có rule nội dung.** 1a tách MỌI trường
  (element, background, photo/art_style, lighting, aesthetics) thành mệnh đề tiếng Việt đã xếp
  hạng. 1b đưa bản phân rã + định nghĩa short/medium/long (độ phủ × độ sâu × loại thông tin)
  cho LLM chọn cả 3 mức trong một lần gọi. Code chỉ KIỂM TRA: chép nguyên văn, lồng nhau
  short ⊆ medium ⊆ long, trần từ (prompt ghi short 16 / medium 50, code chấp nhận tới 20 / 55,
  không tính chủ đề chính); sai thì gọi lại kèm danh sách lỗi (`--max_fix_attempts`, mặc định 2).
  Checklist = chủ đề chính + các mệnh đề đã chọn, và 2a chỉ nhận đúng các mệnh đề đó.
- **Chủ đề chính bám high_level_description, tên file chỉ là gợi ý.** 1a đọc HLD rút ra
  `chu_de_chinh` (1-3 mệnh đề: bức ảnh VỀ CÁI GÌ — vd "nhóm người làm gốm", không phải "hai
  phụ nữ và một người đàn ông"). Gợi ý từ tên thư mục + `common/topic_terms.json` (thuật ngữ
  có dấu, đồng nghĩa) chỉ để 1a chọn cách gọi tên; ảnh không thể hiện chủ đề gợi ý thì 1a bỏ
  qua (`khop_goi_y: false`). Code tự chèn chủ đề vào cả 3 mức; 2b loại ngay nếu judge báo thiếu
  bất kỳ mệnh đề chủ đề nào (`required_facts`), bất kể `--max_missing`. Sửa `topic_terms.json`
  thì cache 1a của các ảnh có gợi ý thay đổi tự được gọi lại. Chủ đề tối đa 3 mệnh đề / 10 từ;
  sai thì 1a gọi lại kèm danh sách lỗi (`--max_fix_attempts`, mặc định 2), hết lượt mới ghi vào
  `step1a_decompose/failures.json` kèm ĐỦ danh sách lỗi (`errors`) và output cuối của LLM
  (`last_output`) để soi.
- **Hỏng mức nào bỏ mức đó.** Hết lượt sửa mà vẫn lỗi thì 1b giữ các mức tự hợp lệ và lồng
  nhau với nhau (hai mức hợp lệ mà không lồng nhau thì bỏ mức ít chi tiết hơn), ghi lý do vào
  `step1b_subjson/partial.json`. Chỉ ảnh không giữ được mức nào mới vào `failures.json` (kèm
  output cuối của LLM). `--retry_partial` gọi lại riêng các ảnh giữ một phần.
- **Deterministic.** 1a và 1b chạy ở `temperature=0` và được cache. Cache 1b gắn dấu vân tay
  (đầu vào + system prompt) — 1a đổi kết quả hoặc sửa prompt 1b thì ảnh đó tự được gọi lại.
- **Model chấm lọc riêng.** Step 2b ưu tiên `JUDGE_BASE_URL` / `JUDGE_MODEL` nếu `.env`
  có khai báo — nên dùng model khác họ với model sinh để tránh thiên vị.
- **Judge chấm mức quan trọng, rule chỉ chặn phần cốt lõi.** Mỗi mệnh đề THIẾU được judge gắn
  `muc_do`: `cot_loi` (chủ thể, món ăn, trang phục, địa danh, thuật ngữ văn hoá, số lượng / màu
  của chủ thể chính, hành động chính, bối cảnh có bản sắc) hoặc `phu` (lấy nét sâu, ánh sáng ban
  ngày, nhỏ, ở góc dưới bên phải...); không chắc thì chấm `cot_loi`. Code chỉ loại khi thiếu
  mệnh đề chủ đề chính, thiếu `cot_loi` quá `--max_missing` (mặc định 0), hoặc có THÊM thông
  tin. Mệnh đề `phu` được BỎ QUA — user thật không nói những thứ đó, và model được train phải
  tự bổ sung. THÊM vẫn chặt như cũ vì prompt đòi thứ không có trong ảnh sẽ dạy model bỏ qua yêu
  cầu của user. `report.json` ghi `missing_phu_top` và `n_passed_with_missing_phu` để soi lại
  xem judge có nới tay quá không; verdict cũ (danh sách chuỗi, không nhãn) được coi là `cot_loi`.
- **Sửa lại prompt bị loại (`--retry_rejected`).** Mặc định TẮT. Bật lên thì sau step 2b,
  `step2b2_correct.py` đưa đúng lý do bị loại (thiếu `cot_loi` / thừa, KHÔNG nhồi mệnh đề phụ
  đã bỏ qua) cho model sửa lại — CHỈ sửa
  đúng phần bị nêu lỗi, không viết lại từ đầu — rồi chấm lại bằng đúng judge của 2b
  (kể cả ràng buộc chủ đề chính `required_facts`). Đạt thì gộp vào tập đạt (`corrected: true`,
  giữ `original_prompt` để audit); vẫn fail thì mới thật sự loại — **retry đúng 1 lần**,
  không lặp thêm. `step2c_split.py` tự phát hiện và gộp `corrected_passed.jsonl` nếu có,
  không cần cấu hình gì thêm.
- **Nhãn Y phải qua step 2d.** `target_json` trong `step2c_split/` vẫn mang trường
  `id` ở mỗi element -- đó là tay cầm nội bộ do step 0 gán để step 1a/1b tham chiếu
  element, KHÔNG thuộc schema Ideogram 4 (`CaptionVerifier` báo `unknown keys ['id']`).
  `step2d_finalize.py` bóc `id` và khoá lại thứ tự key, ghi ra `step2d_final/`.
  **Train bằng `step2d_final/`, không phải `step2c_split/`.**
- **Tắt hẳn cổng lọc (`--ignore_judge`).** Không gọi judge, mọi prompt được coi là đạt
  — không chấm gì cả. Dùng khi debug hoặc muốn tin thẳng đầu ra của 2a.

## Test

```bash
python3 tests/test_pipeline.py       # 146 test, không chạm mạng
```

Logic thuần được test đầy đủ hành vi. Phần gọi model chỉ test được những gì test
được mà không cần server: dựng endpoint/payload/headers, parse response
(kể cả khi bị bọc ```` ```json ````), validate đầu ra của model, dựng messages.

Chưa có server thì vẫn soi được đầu vào của 1b bằng dữ liệu giả (1b gọi model nên chỉ
chạy được `--dry_run`):

```bash
python3 step0_normalize.py --in_dir DATA --out_root test_1 --test --test_samples 20
python3 tools/make_mock_decompose.py \
    --targets_file test_1/step0_normalized/targets.jsonl \
    --out_file     test_1/step1a_decompose/decompose.jsonl
python3 step1b_build_subjson.py --out_root test_1 --dry_run
```

Thống kê prompt có chứa thuật ngữ trong tên file không (`am_tich_000755` → "ấm tích"), trên
cả train/val/test. Phân biệt prompt làm mất thuật ngữ mà JSON gốc có (lỗi pipeline) với ảnh
mà JSON gốc cũng không có thuật ngữ đó:

```bash
python3 tools/check_filename_term.py --out_root test_6
# -> test_6/step2d_final/filename_term_report/{report.json, missing.jsonl}
```

## Hai điểm thiết kế dễ hiểu nhầm

**1. `sub_json` không bao giờ là nhãn huấn luyện.**
Cả 3 mức short/medium/long đều dùng chung nhãn `Y = target_json` đầy đủ.
Prompt ngắn nhưng nhãn vẫn giàu → model buộc phải học cách lấp đầy khoảng trống.
`sub_json` chỉ có hai việc: làm đầu vào cho step 2a, và làm checklist chấm điểm.

**2. Mức chi tiết định nghĩa theo nội dung, không theo số từ.**
Short / medium / long khác nhau ở *độ phủ* (nói tới bao nhiêu phần của ảnh), *độ sâu*
(mỗi phần tả kỹ tới đâu) và *loại thông tin* (chủ thể → thuộc tính → bối cảnh → phong
cách). Số từ chỉ là trần phụ. Định nghĩa đầy đủ nằm trong `STEP1B_SYSTEM`.

## Ngoài phạm vi

Prompt sinh ra ở đây đều **viết đúng chính tả**. Việc chịu được đầu vào sai chính tả,
mất dấu, viết tắt thuộc về một **mô-đun correction riêng** đặt *trước* prompt enhancer.
Trộn nhiễu vào dữ liệu huấn luyện của bước này sẽ làm nhoè mất nhiệm vụ chính
là chi tiết hoá.
