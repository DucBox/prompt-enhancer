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

| Step | File | Gọi model | Đầu ra |
|------|------|:---------:|--------|
| 0  | `step0_normalize.py`       | –   | `output/step0_normalized/` |
| 1a | `step1a_decompose.py`      | ✅  | `output/step1a_decompose/` |
| 1b | `step1b_build_subjson.py`  | –   | `output/step1b_subjson/` |
| 2a | `step2a_verbalize.py`      | ✅  | `output/step2a_prompts/` |
| 2b | `step2b_filter.py`         | ✅  | `output/step2b_filtered/` |
| 2c | `step2c_split.py`          | –   | `output/step2c_split/` |

Mỗi step ghi vào một thư mục riêng, và đọc đầu ra của step trước.

## Chạy

Mọi step đều có `--test` (chạy trên một ít mẫu ngẫu nhiên) và `--test_samples N`:

```bash
# thử nghiệm 20 mẫu
python step0_normalize.py --in_dir ../../data/raw --test --test_samples 20
python step1a_decompose.py --test --test_samples 20
python step1b_build_subjson.py --test --test_samples 20
python step2a_verbalize.py --test --test_samples 20
python step2b_filter.py --test --test_samples 20
python step2c_split.py --test --test_samples 20

# chạy toàn bộ
bash run_all.sh ../../data/raw
```

Hai step gọi model còn có `--dry_run`: dựng messages và in ra màn hình mà **không**
gọi mạng — dùng để kiểm tra prompt trước khi đốt tiền.

### Ghi chú vận hành

- **Cache & resume.** Step 1a / 2a / 2b lưu kết quả từng mẫu ra file riêng. Chạy lại
  sẽ bỏ qua mẫu đã xong, nên có thể dừng giữa chừng rồi chạy tiếp. Dùng `--overwrite`
  để gọi lại từ đầu.
- **Song song.** `--workers N` (mặc định 8).
- **Deterministic.** Step 1a chạy ở `temperature=0` và được cache; mọi việc cắt dữ liệu
  ở step 1b là code thuần có seed. Nghĩa là tái tạo lại y hệt dataset mà không cần gọi lại model.
- **Model chấm lọc riêng.** Step 2b ưu tiên `JUDGE_BASE_URL` / `JUDGE_MODEL` nếu `.env`
  có khai báo — nên dùng model khác họ với model sinh để tránh thiên vị.

## Test

```bash
python tests/test_pipeline.py        # 64 test, không chạm mạng
```

Logic thuần được test đầy đủ hành vi. Phần gọi model chỉ test được những gì test
được mà không cần server: dựng endpoint/payload/headers, parse response
(kể cả khi bị bọc ```` ```json ````), validate đầu ra của model, dựng messages.

Chưa có server thì vẫn chạy thử được đường ống 1b → 2c bằng dữ liệu giả:

```bash
python tools/make_mock_decompose.py --limit 100
python step1b_build_subjson.py --decompose_file output/_mock/decompose.jsonl
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
