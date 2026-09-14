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
#   --from STEP         Bắt đầu từ step này: 0|1a|1b|2a|2b|2b2|2c|2d    (mặc định: 0)
#   --retry_rejected    Bật step 2b2: sửa lại (retry ĐÚNG 1 lần) prompt bị 2b loại
#                        thay vì bỏ trắng. Mặc định TẮT -- không bật thì rejected.jsonl
#                        của 2b vẫn là danh sách cuối cùng bị loại, y như trước.
#   --ignore_judge       Tắt hẳn cổng lọc 2b: KHÔNG gọi judge, coi mọi prompt là đạt,
#                        không chấm gì cả. Dùng khi debug hoặc tin thẳng đầu ra của 2a.
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
RETRY_REJECTED=0
IGNORE_JUDGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in_dir)          IN_DIR="$2"; shift 2 ;;
    --out_dir)         OUT_DIR="$2"; shift 2 ;;
    --test)            TEST=1; shift ;;
    --test_samples)    TEST_SAMPLES="$2"; shift 2 ;;
    --workers)         WORKERS="$2"; shift 2 ;;
    --from)            FROM="$2"; shift 2 ;;
    --retry_rejected)  RETRY_REJECTED=1; shift ;;
    --ignore_judge)    IGNORE_JUDGE=1; shift ;;
    -h|--help)         sed -n '2,23p' "$0"; exit 0 ;;
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

STEP_ORDER=(0 1a 1b 2a 2b 2b2 2c 2d)

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
echo "retry 2b2: $([[ $RETRY_REJECTED -eq 1 ]] && echo BẬT || echo tắt)"
echo "judge 2b : $([[ $IGNORE_JUDGE -eq 1 ]] && echo "BỎ QUA (ignore_judge)" || echo "bật (chấm bình thường)")"

START=$(date +%s)

if should_run 0;  then banner "STEP 0   chuẩn hoá & audit"
  python3 step0_normalize.py --in_dir "$IN_DIR" "${ROOT[@]}" "${TEST_ARGS[@]+"${TEST_ARGS[@]}"}"; fi

if should_run 1a; then banner "STEP 1a  gom nhóm & phân rã mệnh đề  [gọi model]"
  python3 step1a_decompose.py "${ROOT[@]}" --workers "$WORKERS"; fi

if should_run 1b; then banner "STEP 1b  chọn lọc short/medium/long  [gọi model]"
  python3 step1b_build_subjson.py "${ROOT[@]}" --workers "$WORKERS"; fi

if should_run 2a; then banner "STEP 2a  sinh user prompt  [gọi model]"
  python3 step2a_verbalize.py "${ROOT[@]}" --workers "$WORKERS"; fi

IGNORE_JUDGE_ARGS=()
if [[ $IGNORE_JUDGE -eq 1 ]]; then IGNORE_JUDGE_ARGS=(--ignore_judge); fi

if should_run 2b; then banner "STEP 2b  lọc chất lượng  [gọi model]"
  python3 step2b_filter.py "${ROOT[@]}" --workers "$WORKERS" \
    "${IGNORE_JUDGE_ARGS[@]+"${IGNORE_JUDGE_ARGS[@]}"}"; fi

if should_run 2b2 && [[ $RETRY_REJECTED -eq 1 ]]; then
  banner "STEP 2b2  sửa lại prompt bị loại (retry 1 lần)  [gọi model]"
  python3 step2b2_correct.py "${ROOT[@]}" --workers "$WORKERS"
fi

if should_run 2c; then banner "STEP 2c  chia tập"
  python3 step2c_split.py "${ROOT[@]}"; fi

if should_run 2d; then banner "STEP 2d  chuẩn hoá nhãn Y về schema Ideogram 4"
  python3 step2d_finalize.py "${ROOT[@]}"; fi

ELAPSED=$(( $(date +%s) - START ))
echo
echo "================================================================"
echo "Xong sau ${ELAPSED}s. Cây đầu ra:"
echo
find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/  /'
echo
echo "Dữ liệu huấn luyện: $OUT_DIR/step2d_final/{train,val,test}.jsonl"
echo "================================================================"
