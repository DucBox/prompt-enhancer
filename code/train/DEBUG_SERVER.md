# Chạy & debug train trên server

Đứng ở `code/train`, venv đã activate. Sửa các biến ở mục 0 nếu đổi đường dẫn.

## 0. Biến dùng chung

```bash
cd /workspace/genai/ducnq7/code/train
export MODEL_PATH=/workspace/genai/anhph53/models/Qwen3.6-27B
export DATA=/workspace/genai/ducnq7/data/sample/test_train/test_5/step2d_final
export OUT=/workspace/genai/ducnq7/runs/method_1/tests/exp3

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # script cũng tự bật khi MODEL_PATH là folder
export PYTHONUNBUFFERED=1                                              # print hiện ngay, không bị buffer
export PYTORCH_ALLOC_CONF=expandable_segments:True                     # giảm phân mảnh VRAM
```

## 1. Cập nhật code

```bash
git pull
git log --oneline -1                                   # phải là commit mới nhất trên main
grep -c dbg_step train_prompt_enhancer_qwen36.py       # > 0 là đã có chế độ --debug
```

## 2. Dọn process cũ

```bash
ps -eo pid,stat,etime,cmd | grep -E "train_prompt|torch.distributed" | grep -v grep
pkill -9 -f train_prompt_enhancer_qwen36.py
nvidia-smi                                             # GPU phải trống
```

## 3. Kiểm tra môi trường

```bash
python check_env.py --backend hf --model_name $MODEL_PATH --data_file $DATA/train.jsonl
```

Độ dài mẫu in ra phải là vài trăm–vài nghìn token (ra `2` là lỗi tokenizer).

## 4. Chẩn đoán khởi động chậm (đứng lâu ở "Setting OMP_NUM_THREADS")

Nếu bật `--debug` mà KHÔNG thấy dòng `[debug r0 +0.0s] debug bật` thì chậm nằm ở lúc Python
khởi động / import, trước khi vào script.

### 4.1 Đo thời gian khởi động và import

```bash
time python -c "pass"
time python -c "import torch"
time python -c "import torch, transformers, peft, bitsandbytes, datasets"
```

Lệnh nào mất hàng chục giây–phút -> ổ chứa venv chậm (thường là ổ mạng `/workspace`).

### 4.2 Thời gian từng module import (kể cả trước khi vào script)

```bash
PYTHONPROFILEIMPORTTIME=1 python -c "import torch, transformers, peft, bitsandbytes, datasets" 2> import_time.log
sort -t'|' -k2 -n import_time.log | tail -25
```

Cột 2 là thời gian tích luỹ (microsecond) của module đó.

### 4.3 Ổ đĩa có phải ổ mạng không

```bash
df -hT /workspace "$VIRTUAL_ENV" $MODEL_PATH
mount | grep -E "nfs|fuse|ceph|lustre" | head
```

### 4.4 Khi đang đứng im: process đang làm gì

Mở terminal khác:

```bash
ps -eo pid,stat,etime,pcpu,cmd | grep train_prompt | grep -v grep
```

* `STAT` = `D` -> đang chờ đọc đĩa (IO chậm).
* `STAT` = `R` + `%CPU` cao -> đang tính toán / import.
* `STAT` = `S` + CPU ~0 -> đang chờ (mạng, NCCL, lock).

Xem đúng dòng đang kẹt:

```bash
pip install py-spy          # nếu chưa có
for pid in $(pgrep -f "train_prompt_enhancer_qwen36.py --model_name"); do
  echo "===== PID $pid ====="; py-spy dump --pid $pid
done
```

* Thấy `site.py` / `importlib` / `<frozen ...>` -> chậm ở import (ổ đĩa).
* Thấy `huggingface_hub` / `requests` / `socket` -> đang gọi mạng (thiếu offline).
* Thấy `nccl` / `all_reduce` / `barrier` -> kẹt giao tiếp giữa các GPU.

### 4.5 Nếu đúng là ổ chậm

```bash
# Làm nóng cache trước khi torchrun
python -c "import torch, transformers, peft, bitsandbytes, datasets" && echo warm

# Hoặc chép model sang ổ local (cần đủ dung lượng, ~55GB)
df -h /dev/shm /tmp
cp -r $MODEL_PATH /dev/shm/Qwen3.6-27B && export MODEL_PATH=/dev/shm/Qwen3.6-27B
```

## 5. Chạy thử 1 process có debug (dễ đọc log nhất)

```bash
CUDA_VISIBLE_DEVICES=0 PE_DEBUG_STACK_EVERY=120 python train_prompt_enhancer_qwen36.py \
  --model_name $MODEL_PATH \
  --train_file $DATA/train.jsonl --eval_file $DATA/val.jsonl \
  --output_dir $OUT-debug1 \
  --mode no_cot --backend hf \
  --per_device_batch_size 1 --global_batch_size 16 --lora_r 16 \
  --epochs 1 --eval_steps 10 --debug 2>&1 | tee debug_1gpu.log
```

* Mỗi bước in `-> bước ...` lúc bắt đầu và `<- bước xong (Ns)` khi xong, kèm VRAM.
* Có `->` mà chưa có `<-` -> đang treo / chậm ở bước đó.
* Mỗi `PE_DEBUG_STACK_EVERY` giây tự in stack mọi luồng (0 = tắt).

## 6. Train nhiều GPU (8 GPU, backend hf)

```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 29533 \
  train_prompt_enhancer_qwen36.py \
  --model_name $MODEL_PATH \
  --train_file $DATA/train.jsonl --eval_file $DATA/val.jsonl \
  --output_dir $OUT \
  --mode no_cot --backend hf \
  --per_device_batch_size 1 --global_batch_size 16 --lora_r 16 \
  --epochs 2 --eval_steps 10 --debug 2>&1 | tee train_8gpu.log
```

* `global_batch_size` nên chia hết cho `per_device_batch_size × 8`; grad accumulation tự tính.
* Chạy ổn rồi thì bỏ `--debug`.
* 463 mẫu: 2–3 epoch là đủ để thử, nhiều hơn dễ overfit.

## 7. Train 1 GPU (Unsloth)

```bash
CUDA_VISIBLE_DEVICES=0 python train_prompt_enhancer_qwen36.py \
  --model_name $MODEL_PATH \
  --train_file $DATA/train.jsonl --eval_file $DATA/val.jsonl \
  --output_dir $OUT-unsloth \
  --mode no_cot --backend unsloth \
  --per_device_batch_size 1 --global_batch_size 16 --epochs 2 --eval_steps 10
```

## 8. Nếu OOM (CUDA out of memory)

Thử lần lượt:

1. `--per_device_batch_size 1` (giữ `--global_batch_size`).
2. `--max_seq_length 3072` (xem `over_max_seq=` trong log xem bị bỏ bao nhiêu mẫu).
3. `--lora_r 16`.
4. `--backend unsloth` 1 GPU (ít VRAM nhất).

Gradient checkpointing đã bật sẵn, không cần chỉnh.

## 9. Tăng tốc: kernel cho linear attention

Log có dòng `The fast path is not available ...` -> chưa có 2 gói dưới đây.

```bash
pip install flash-linear-attention
pip install causal-conv1d --no-build-isolation      # cần nvcc, build vài phút
python -c "import fla, causal_conv1d; print('ok')"
```

Server không ra mạng: tải wheel ở máy có mạng (cùng Python 3.12, Linux x86_64, torch 2.10 cu128):

```bash
# máy có mạng
pip download flash-linear-attention -d wheels/
pip wheel causal-conv1d --no-build-isolation -w wheels/
# copy wheels/ lên server rồi:
pip install --no-index --find-links wheels/ flash-linear-attention causal-conv1d
```

## 10. Cảnh báo "Skipping import of cpp extensions ... torch >= 2.11.0"

Do torchao 0.18.0 cần torch 2.11+. Không ảnh hưởng train (script dùng bitsandbytes). Muốn hết cảnh báo:

```bash
pip show torchao
pip install torchao==0.16.0        # bản có C++ extension cho torch 2.10
# hoặc gỡ hẳn:
pip uninstall -y torchao
python -c "import transformers, peft; print('ok')"
```

## 11. Gửi lại khi cần hỗ trợ

```bash
tail -60 debug_1gpu.log          # hoặc train_8gpu.log
grep "\[debug" train_8gpu.log | tail -30
sort -t'|' -k2 -n import_time.log | tail -25
```
