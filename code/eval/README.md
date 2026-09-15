# Đánh giá (bước 5)

## 1. Infer -> predictions

`infer.py` chạy tập test bằng model gốc và/hoặc model gốc + LoRA adapter. Bật cờ nào chạy cờ đó:

| Cờ | Chạy |
|---|---|
| `--run_baseline` | model gốc, không adapter |
| `--run_adapter` | model gốc + `--adapter_dir` |
| cả hai | cả hai, chỉ load model 1 lần (baseline = tắt adapter tạm thời) |

Prompt dựng giống hệt lúc train (`--mode no_cot`): system prompt theo mức của từng dòng, lấy từ
`<adapter_dir>/system_prompt_<mức>.txt`; baseline dùng cùng system prompt đó. Sinh greedy.

```bash
cd /workspace/genai/ducnq7/code/eval
export MODEL_PATH=/workspace/genai/anhph53/models/Qwen3.6-27B
export ADAPTER=/workspace/genai/ducnq7/runs/method_1/tests/exp4/final_adapter
export DATA=/workspace/genai/ducnq7/data/sample/test_train/test_5/step2d_final
export EVAL_OUT=/workspace/genai/ducnq7/runs/method_1/tests/exp4/eval
export PYTHONUNBUFFERED=1

# Thử nhanh 3 dòng, 1 GPU, chỉ adapter
CUDA_VISIBLE_DEVICES=0 python infer.py --model_name $MODEL_PATH --adapter_dir $ADAPTER \
  --test_file $DATA/test.jsonl --output_dir $EVAL_OUT-smoke --run_adapter --limit 3

# Chạy thật: 8 GPU, cả baseline lẫn adapter
python -m torch.distributed.run --nproc_per_node=8 --master_port 29544 infer.py \
  --model_name $MODEL_PATH --adapter_dir $ADAPTER \
  --test_file $DATA/test.jsonl --output_dir $EVAL_OUT \
  --run_baseline --run_adapter 2>&1 | tee infer.log
```

Tuỳ chọn khác:

* `--modes short long` — chỉ chạy các mức này.
* `--limit N` — chỉ N dòng đầu.
* `--max_new_tokens 4096` — tăng nếu log báo `CHẠM max_new_tokens`.
* `--batch_size 1` — tăng để nhanh hơn (padding trái); 1 an toàn nhất.
* `--no-load_in_4bit` — base bf16 thay vì 4-bit như lúc train (27B bf16 cần > 1 GPU 40GB, chạy 1 process).
* `--overwrite` — sinh lại cả dòng đã có cache. Mặc định chạy lại chỉ sinh các dòng còn thiếu.

Đầu ra trong `--output_dir`:

```
raw/<adapter|baseline>/<id>__<mức>.json   cache từng dòng
predictions_adapter.jsonl                 gộp theo thứ tự tập test
predictions_baseline.jsonl
run_config.json                           tham số + nguồn system prompt
```

Mỗi dòng prediction: `id, mode, variant, user_prompt, raw_output, pred_json` (null nếu không parse
được), `parse_error, n_input_tokens, n_output_tokens, hit_max_new_tokens, gen_seconds,
target_json, sub_json`.

Test (không cần GPU): `python -m unittest discover -s tests`
