#!/usr/bin/env bash
# Chạy toàn bộ pipeline method_1.
#
#   bash run_all.sh <thư_mục_data_gốc> [--test]
#
# Thêm --test để chạy thử trên 20 mẫu ngẫu nhiên ở mọi step.
set -euo pipefail

IN_DIR="${1:?Thiếu tham số: thư mục chứa file .txt gốc}"
shift || true
EXTRA="$*"
WORKERS="${WORKERS:-8}"

cd "$(dirname "$0")"

echo "==> STEP 0  chuẩn hoá & audit"
python3 step0_normalize.py --in_dir "$IN_DIR" $EXTRA

echo "==> STEP 1a  gom nhóm & phân rã mệnh đề  [gọi model]"
python3 step1a_decompose.py --workers "$WORKERS" $EXTRA

echo "==> STEP 1b  nén 2 trục"
python3 step1b_build_subjson.py $EXTRA

echo "==> STEP 2a  sinh user prompt  [gọi model]"
python3 step2a_verbalize.py --workers "$WORKERS" $EXTRA

echo "==> STEP 2b  lọc chất lượng  [gọi model]"
python3 step2b_filter.py --workers "$WORKERS" $EXTRA

echo "==> STEP 2c  chia tập"
python3 step2c_split.py $EXTRA

echo
echo "Xong. Dữ liệu huấn luyện: output/step2c_split/{train,val,test}.jsonl"
