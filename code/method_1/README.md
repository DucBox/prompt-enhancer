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
| 1b | `step1b_build_subjson.py`  | –   | `step1b_subjson/` |
| 2a | `step2a_verbalize.py`      | ✅  | `step2a_prompts/` |
| 2b | `step2b_filter.py`         | ✅  | `step2b_filtered/` |
| 2c | `step2c_split.py`          | –   | `step2c_split/` |

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
├── step1b_subjson/       subjson.jsonl, stats.json
├── step2a_prompts/       prompts.jsonl, prompts/<id>__<level>.json
├── step2b_filtered/      passed.jsonl, rejected.jsonl, report.json
└── step2c_split/         train.jsonl, val.jsonl, test.jsonl
```

| Cờ của `run_all.sh` | Mặc định | |
|---|---|---|
| `--in_dir DIR` | – | thư mục data gốc (bắt buộc trừ khi dùng `--from`) |
| `--out_dir DIR` | `output` | thư mục output gốc |
| `--test` | tắt | chạy thử trên ít mẫu |
| `--test_samples N` | 20 | |
| `--workers N` | 4 | số luồng gọi model |
| `--from STEP` | `0` | bắt đầu từ `0\|1a\|1b\|2a\|2b\|2c` |

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

- **Cache & resume.** Step 1a / 2a / 2b lưu kết quả từng mẫu ra file riêng. Chạy lại
  sẽ bỏ qua mẫu đã xong, nên có thể dừng giữa chừng rồi chạy tiếp. Dùng `--overwrite`
  để gọi lại từ đầu.
- **Song song.** `--workers N` (mặc định 4).
- **Deterministic.** Step 1a chạy ở `temperature=0` và được cache; mọi việc cắt dữ liệu
  ở step 1b là code thuần có seed. Nghĩa là tái tạo lại y hệt dataset mà không cần gọi lại model.
- **Model chấm lọc riêng.** Step 2b ưu tiên `JUDGE_BASE_URL` / `JUDGE_MODEL` nếu `.env`
  có khai báo — nên dùng model khác họ với model sinh để tránh thiên vị.

## Test

```bash
python3 tests/test_pipeline.py       # 73 test, không chạm mạng
```

Logic thuần được test đầy đủ hành vi. Phần gọi model chỉ test được những gì test
được mà không cần server: dựng endpoint/payload/headers, parse response
(kể cả khi bị bọc ```` ```json ````), validate đầu ra của model, dựng messages.

Chưa có server thì vẫn chạy thử được đường ống 1b → 2c bằng dữ liệu giả:

```bash
python3 step0_normalize.py --in_dir DATA --out_root test_1 --test --test_samples 20
python3 tools/make_mock_decompose.py \
    --targets_file test_1/step0_normalized/targets.jsonl \
    --out_file     test_1/step1a_decompose/decompose.jsonl
python3 step1b_build_subjson.py --out_root test_1
```

## Hai điểm thiết kế dễ hiểu nhầm

**1. `sub_json` không bao giờ là nhãn huấn luyện.**
Cả 3 mức short/medium/long đều dùng chung nhãn `Y = target_json` đầy đủ.
Prompt ngắn nhưng nhãn vẫn giàu → model buộc phải học cách lấp đầy khoảng trống.
`sub_json` chỉ có hai việc: làm đầu vào cho step 2a, và làm checklist chấm điểm.

**2. Nén theo hai trục, không phải một.**
Chỉ cắt bớt số element là chưa đủ — mỗi `desc` dài trung bình 19 từ, nên prompt
"short" giữ 2 element vẫn ra ~38 từ, đặc như JSON. Phải cắt cả *bề rộng*
(giữ bao nhiêu nhóm khái niệm) lẫn *chiều sâu* (mỗi element giữ bao nhiêu mệnh đề).

## Ngoài phạm vi

Prompt sinh ra ở đây đều **viết đúng chính tả**. Việc chịu được đầu vào sai chính tả,
mất dấu, viết tắt thuộc về một **mô-đun correction riêng** đặt *trước* prompt enhancer.
Trộn nhiễu vào dữ liệu huấn luyện của bước này sẽ làm nhoè mất nhiệm vụ chính
là chi tiết hoá.
