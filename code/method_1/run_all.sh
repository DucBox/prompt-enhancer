#!/usr/bin/env bash
#
# Chạy toàn bộ pipeline method_1.
#
#   bash run_all.sh --in_dir DATA --out_dir test_1 --test
#   bash run_all.sh --in_dir DATA --out_dir run_full --workers 4
#
# Tham số:
#   --in_dir DIR        Thư mục chứa file .txt/.json gốc                (bắt buộc)
#   --out_dir DIR       Thư mục output gốc; mỗi step một thư mục con    (mặc định: output)
#   --test              Chạy thử trên một ít mẫu ngẫu nhiên
#   --test_samples N    Số mẫu khi bật --test                           (mặc định: 20)
#   --workers N         Số luồng gọi model song song                    (mặc định: 4)
#   --from STEP         Bắt đầu từ step này: 0|1a|1b|2a|2b|2c           (mặc định: 0)
#
# Ghi chú về --test: cờ này CHỈ đặt ở step 0. Step 0 lọc dữ liệu còn N mẫu,
# các step sau tự kế thừa N mẫu đó. Nếu đặt --test ở mọi step thì mỗi step lại
# bốc ngẫu nhiên tiếp và chuỗi dữ liệu bị đứt.

set -euo pipefail
cd "$(dirname "$0")"

IN_DIR=""
OUT_DIR="output"
TEST=0
TEST_SAMPLES=20
WORKERS=4
FROM="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in_dir)        IN_DIR="$2"; shift 2 ;;
    --out_dir)       OUT_DIR="$2"; shift 2 ;;
    --test)          TEST=1; shift ;;
    --test_samples)  TEST_SAMPLES="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --from)          FROM="$2"; shift 2 ;;
    -h|--help)       sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Tham số lạ: $1  (dùng --help)" >&2; exit 1 ;;
  esac
done

if [[ -z "$IN_DIR" && "$FROM" == "0" ]]; then
  echo "Thiếu --in_dir. Xem: bash run_all.sh --help" >&2
  exit 1
fi

# Chỉ step 0 nhận cờ test; các step sau kế thừa dữ liệu đã lọc.
TEST_ARGS=()
if [[ $TEST -eq 1 ]]; then
  TEST_ARGS=(--test --test_samples "$TEST_SAMPLES")
fi

ROOT=(--out_root "$OUT_DIR")

STEP_ORDER=(0 1a 1b 2a 2b 2c)

step_index() {   # step_index <step> -> vị trí trong STEP_ORDER, hoặc -1
  local i
  for i in "${!STEP_ORDER[@]}"; do
    [[ "${STEP_ORDER[$i]}" == "$1" ]] && { echo "$i"; return; }
  done
  echo "-1"
}

FROM_IDX=$(step_index "$FROM")
if [[ "$FROM_IDX" == "-1" ]]; then
  echo "--from không hợp lệ: '$FROM' (chọn: ${STEP_ORDER[*]})" >&2
  exit 1
fi

should_run() {   # chạy nếu step đứng từ FROM trở đi
  [[ $(step_index "$1") -ge $FROM_IDX ]]
}

banner() { echo; echo "################  $*  ################"; }

echo "in_dir   : ${IN_DIR:-(bỏ qua step 0)}"
echo "out_dir  : $OUT_DIR/"
echo "workers  : $WORKERS"
if [[ $TEST -eq 1 ]]; then
  echo "chế độ   : TEST — $TEST_SAMPLES mẫu ngẫu nhiên"
else
  echo "chế độ   : FULL — toàn bộ dữ liệu"
fi

START=$(date +%s)

if should_run 0;  then banner "STEP 0   chuẩn hoá & audit"
  python3 step0_normalize.py --in_dir "$IN_DIR" "${ROOT[@]}" "${TEST_ARGS[@]+"${TEST_ARGS[@]}"}"; fi

if should_run 1a; then banner "STEP 1a  gom nhóm & phân rã mệnh đề  [gọi model]"
  python3 step1a_decompose.py "${ROOT[@]}" --workers "$WORKERS"; fi

if should_run 1b; then banner "STEP 1b  nén 2 trục"
  python3 step1b_build_subjson.py "${ROOT[@]}"; fi

if should_run 2a; then banner "STEP 2a  sinh user prompt  [gọi model]"
  python3 step2a_verbalize.py "${ROOT[@]}" --workers "$WORKERS"; fi

if should_run 2b; then banner "STEP 2b  lọc chất lượng  [gọi model]"
  python3 step2b_filter.py "${ROOT[@]}" --workers "$WORKERS"; fi

if should_run 2c; then banner "STEP 2c  chia tập"
  python3 step2c_split.py "${ROOT[@]}"; fi

ELAPSED=$(( $(date +%s) - START ))
echo
echo "================================================================"
echo "Xong sau ${ELAPSED}s. Cây đầu ra:"
echo
find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/  /'
echo
echo "Dữ liệu huấn luyện: $OUT_DIR/step2c_split/{train,val,test}.jsonl"
echo "================================================================"
