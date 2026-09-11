"""Toàn bộ system prompt + few-shot của pipeline.

Gom một chỗ để sửa prompt không phải đụng vào logic từng step.
Ba prompt tương ứng ba lần gọi model:
    STEP1A  phân rã target_json thành nhóm khái niệm + mệnh đề nguyên tử
    STEP2A  viết user prompt từ sub_json (giọng người dùng thật)
    STEP2B  kiểm tra prompt có thêm/thiếu so với sub_json không
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# =============================================================================
# STEP 1a — gom nhóm khái niệm & phân rã mệnh đề
# =============================================================================

STEP1A_SYSTEM = r"""Bạn là công cụ phân tích mô tả ảnh. Đầu vào là một JSON mô tả chi tiết một bức ảnh.
Nhiệm vụ của bạn gồm HAI phần. Chỉ trả về đúng một JSON object, không giải thích gì thêm.

=== PHẦN 1: GOM NHÓM KHÁI NIỆM (concept_groups) ===
Gom các element thành những "khái niệm" mà một NGƯỜI DÙNG BÌNH THƯỜNG sẽ nhắc tới như một thứ duy nhất.

Luật gom nhóm:
- Nhiều người cùng xuất hiện trong một cảnh → gom thành MỘT nhóm.
  Ví dụ: 3 element "nam áo trắng", "nam áo đen", "nữ áo dài" → một nhóm "nhóm bạn", cardinality "exact:3".
- Nhiều vật cùng loại → gom thành MỘT nhóm ("hai quả trứng cút", "ba chiếc thuyền").
- Địa danh, công trình, món ăn, trang phục mang bản sắc Việt Nam → LUÔN là một nhóm RIÊNG,
  đặt "cultural": true (ví dụ: Lăng Bác, chợ nổi Cái Răng, áo dài, khăn xếp, bánh tét, nón lá, ấm tích).
- Các chi tiết nền phụ (cây cối, mây, tường, sàn, đồ vặt) gom vào nhóm "cảnh nền".

Trường cardinality CHỈ có hai dạng:
- "exact:N" khi đếm được chính xác N cá thể (mọi thành viên đều là cá thể đơn lẻ).
- "vague"   khi mô tả dùng lượng từ mơ hồ: several, numerous, multiple, a crowd of, a cluster of, many...
QUAN TRỌNG: TUYỆT ĐỐI KHÔNG bịa ra con số chính xác khi mô tả gốc nói mơ hồ. Sai chỗ này là lỗi nặng nhất.

Xếp các nhóm theo thứ tự QUAN TRỌNG GIẢM DẦN. Nhóm quan trọng nhất là nhóm được nhắc tới
trong high_level_description và là chủ thể chính của bức ảnh.

=== PHẦN 2: PHÂN RÃ MỆNH ĐỀ (facts) ===
Với MỖI element, tách phần mô tả thành các mệnh đề nhỏ nhất, mỗi mệnh đề chỉ nói MỘT ý.

- Mệnh đề đầu tiên (rank 0) LUÔN là danh từ chủ thể trần trụi, viết bằng TIẾNG VIỆT ngắn gọn.
- Các mệnh đề sau xếp theo mức độ quan trọng giảm dần: thứ mà người dùng nhiều khả năng
  nhắc tới nhất đứng trước (màu sắc, chất liệu, công dụng thường quan trọng hơn vị trí trong khung hình).
- Mỗi mệnh đề viết bằng tiếng Việt, ngắn (2-6 từ).
- Giữ NGUYÊN VĂN tên riêng và thuật ngữ văn hoá Việt.
- Loại "kind" thuộc: subject | color | material | attribute | action | position | count

=== ĐỊNH DẠNG ĐẦU RA ===
{
  "concept_groups": [
    {"name": "...", "member_ids": [0,1], "cardinality": "exact:2", "cultural": false}
  ],
  "facts": {
    "0": [
      {"rank": 0, "kind": "subject", "text": "..."},
      {"rank": 1, "kind": "color", "text": "..."}
    ]
  }
}
Khoá của "facts" là id của element, dạng chuỗi. Mọi element đều phải có mặt."""


STEP1A_FEWSHOT_INPUT = {
    "high_level_description": (
        "A top-down aerial photograph of Chợ nổi Cái Răng (Cai Rang Floating Market), showing "
        "three wooden boats clustered on a wide, muddy brown river."
    ),
    "style_description": {
        "aesthetics": "documentary, vibrant",
        "lighting": "bright daylight",
        "photo": "top-down aerial, wide shot",
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "The surface of a wide, murky brown river fills the frame.",
        "elements": [
            {"id": 0, "type": "obj", "desc": "A long purple wooden boat positioned horizontally across the center of the frame, serving as a floating restaurant with cooking equipment and passengers aboard."},
            {"id": 1, "type": "obj", "desc": "Several people visible across the boats: vendors wearing conical hats preparing food on the purple boat."},
            {"id": 2, "type": "obj", "desc": "A bright red plastic bucket sitting on the deck of the purple boat near the right side."},
            {"id": 3, "type": "obj", "desc": "A long blue wooden boat positioned at the bottom of the frame, carrying several seated passengers."},
        ],
    },
}

STEP1A_FEWSHOT_OUTPUT = {
    "concept_groups": [
        {"name": "chợ nổi Cái Răng", "member_ids": [], "cardinality": "exact:1", "cultural": True},
        {"name": "những chiếc thuyền gỗ", "member_ids": [0, 3], "cardinality": "exact:2", "cultural": False},
        {"name": "người bán hàng đội nón lá", "member_ids": [1], "cardinality": "vague", "cultural": True},
        {"name": "cảnh nền", "member_ids": [2], "cardinality": "vague", "cultural": False},
    ],
    "facts": {
        "0": [
            {"rank": 0, "kind": "subject", "text": "chiếc thuyền"},
            {"rank": 1, "kind": "color", "text": "màu tím"},
            {"rank": 2, "kind": "material", "text": "bằng gỗ"},
            {"rank": 3, "kind": "attribute", "text": "là quán ăn nổi"},
            {"rank": 4, "kind": "attribute", "text": "dài"},
            {"rank": 5, "kind": "position", "text": "nằm ngang giữa khung hình"},
            {"rank": 6, "kind": "attribute", "text": "có khách trên thuyền"},
        ],
        "1": [
            {"rank": 0, "kind": "subject", "text": "những người bán hàng"},
            {"rank": 1, "kind": "attribute", "text": "đội nón lá"},
            {"rank": 2, "kind": "action", "text": "đang nấu đồ ăn"},
        ],
        "2": [
            {"rank": 0, "kind": "subject", "text": "cái xô nhựa"},
            {"rank": 1, "kind": "color", "text": "màu đỏ"},
            {"rank": 2, "kind": "position", "text": "trên boong thuyền"},
        ],
        "3": [
            {"rank": 0, "kind": "subject", "text": "chiếc thuyền"},
            {"rank": 1, "kind": "color", "text": "màu xanh"},
            {"rank": 2, "kind": "material", "text": "bằng gỗ"},
            {"rank": 3, "kind": "attribute", "text": "chở khách ngồi thành hàng"},
        ],
    },
}


def build_step1a_messages(target_json: Dict[str, Any]) -> List[Dict[str, str]]:
    dumps = lambda o: json.dumps(o, ensure_ascii=False, indent=2)  # noqa: E731
    return [
        {"role": "system", "content": STEP1A_SYSTEM},
        {"role": "user", "content": dumps(STEP1A_FEWSHOT_INPUT)},
        {"role": "assistant", "content": dumps(STEP1A_FEWSHOT_OUTPUT)},
        {"role": "user", "content": dumps(target_json)},
    ]


# =============================================================================
# STEP 2a — viết user prompt từ sub_json
# =============================================================================

STEP2A_SYSTEM = r"""Bạn đang đóng vai NGƯỜI DÙNG muốn nhờ AI tạo ra một tấm ảnh.
Bạn được cho một bản mô tả có cấu trúc. Hãy viết ra CÂU YÊU CẦU của bạn.
Chỉ trả về đúng câu yêu cầu đó, không thêm lời dẫn, không giải thích, không đặt trong ngoặc kép.

BẮT BUỘC
- Viết như người ĐANG ĐẶT HÀNG một bức ảnh chưa tồn tại,
  KHÔNG phải người đang mô tả một bức ảnh có sẵn trước mặt.
- Chỉ nói những gì có trong bản mô tả. Cấm thêm bất kỳ thông tin mới nào.
- Giữ NGUYÊN VĂN thuật ngữ tiếng Việt (áo dài, khăn xếp, bánh tét, chợ nổi Cái Răng, nón lá...).
- Nếu bản mô tả ghi số lượng chính xác thì phải nói đúng số đó.
  Nếu ghi mơ hồ ("nhiều", "mấy") thì cũng nói mơ hồ, KHÔNG được tự chế ra con số.

CẤM
- Cấm liệt kê từng vật thể thành từng mệnh đề riêng biệt.
  Hãy gộp các vật cùng loại lại: "ba chiếc thuyền gỗ", "mấy người đội nón lá".
- Cấm dùng ngôn ngữ kỹ thuật về khung hình: "nằm ở góc trên bên trái",
  "chiếm phần dưới khung hình", "ở tiền cảnh", "trong hậu cảnh".
  Người dùng thật nói: "phía sau", "bên cạnh", "ở giữa", "đằng xa".
- Cấm bám theo thứ tự của bản mô tả. Hãy bắt đầu bằng thứ bạn quan tâm nhất.
- Cấm văn dịch. Viết như người Việt nói chuyện bình thường.
- Cấm lặp lại một khuôn mở đầu; hãy đa dạng cách vào câu.
- Cấm dùng dấu đầu dòng, gạch đầu dòng hay xuống dòng. Viết liền thành câu/đoạn.
- Viết ĐÚNG CHÍNH TẢ, đủ dấu tiếng Việt. Không mô phỏng lỗi gõ hay viết tắt."""

# Few-shot viết tay: 3 mức độ dài + 1 ví dụ short thứ hai để tránh model bám một khuôn mở đầu.
# Tất cả đều viết đúng chính tả — dữ liệu nhiễu không thuộc phạm vi bước này.
STEP2A_FEWSHOT = [
    (
        {
            "detail_level": "short",
            "length_hint": "8-20 từ",
            "persona": {"vai": "user phổ thông", "giọng": "mô tả trung tính",
                        "ngôn ngữ": "tiếng Việt"},
            "scene": "chụp từ trên cao",
            "groups": [
                {"name": "chợ nổi Cái Răng", "cardinality": "exact:1", "facts": ["chợ nổi Cái Răng"]},
                {"name": "những chiếc thuyền gỗ", "cardinality": "vague",
                 "facts": ["chiếc thuyền", "bằng gỗ"]},
            ],
        },
        "Ảnh chụp từ trên cao chợ nổi Cái Răng với mấy chiếc thuyền gỗ",
    ),
    (
        {
            "detail_level": "medium",
            "length_hint": "30-60 từ",
            "persona": {"vai": "designer", "giọng": "ra lệnh",
                        "ngôn ngữ": "tiếng Việt"},
            "scene": "chụp từ flycam, sông nước đục",
            "groups": [
                {"name": "chợ nổi Cái Răng", "cardinality": "exact:1", "facts": ["chợ nổi Cái Răng"]},
                {"name": "những chiếc thuyền gỗ", "cardinality": "exact:3",
                 "facts": ["chiếc thuyền", "bằng gỗ", "một chiếc màu tím", "là quán ăn nổi"]},
                {"name": "người bán hàng", "cardinality": "vague",
                 "facts": ["những người bán hàng", "đội nón lá", "đang nấu đồ ăn"]},
            ],
        },
        "Chụp từ flycam chợ nổi Cái Răng trên sông nước đục, ba chiếc thuyền gỗ neo gần nhau, "
        "một chiếc màu tím bán đồ ăn, mấy người đội nón lá đang nấu nướng.",
    ),
    (
        {
            "detail_level": "long",
            "length_hint": "100-200 từ",
            "persona": {"vai": "user phổ thông", "giọng": "kể lể lan man",
                        "ngôn ngữ": "tiếng Việt"},
            "scene": "chụp từ trên cao, mặt sông rộng màu nâu đục",
            "groups": [
                {"name": "chợ nổi Cái Răng", "cardinality": "exact:1", "facts": ["chợ nổi Cái Răng"]},
                {"name": "thuyền quán ăn", "cardinality": "exact:1",
                 "facts": ["chiếc thuyền", "màu tím", "bằng gỗ", "dài", "là quán ăn nổi"]},
                {"name": "người bán hàng", "cardinality": "vague",
                 "facts": ["những người bán hàng", "đội nón lá", "đang nấu đồ ăn"]},
                {"name": "thuyền chở hàng", "cardinality": "exact:1",
                 "facts": ["chiếc thuyền", "chở sọt trái cây"]},
                {"name": "thuyền chở khách", "cardinality": "exact:1",
                 "facts": ["chiếc thuyền", "màu xanh", "chở khách ngồi thành hàng"]},
            ],
        },
        "Mình muốn ảnh chụp từ trên cao cảnh chợ nổi Cái Răng. Mặt sông rộng màu nâu đục chiếm gần "
        "hết khung hình. Ở giữa là chiếc thuyền gỗ dài màu tím làm quán ăn nổi, có người đội nón lá "
        "đang nấu. Bên trên là thuyền chở đầy sọt trái cây, phía dưới là chiếc thuyền xanh chở khách "
        "ngồi thành hàng.",
    ),
    (
        {
            "detail_level": "short",
            "length_hint": "8-20 từ",
            "persona": {"vai": "user phổ thông", "giọng": "ra lệnh",
                        "ngôn ngữ": "tiếng Việt"},
            "scene": "chụp trong studio",
            "groups": [
                {"name": "áo dài nam", "cardinality": "exact:1",
                 "facts": ["áo dài nam", "màu đỏ"]},
                {"name": "khăn xếp", "cardinality": "exact:1",
                 "facts": ["khăn xếp", "màu đen"]},
            ],
        },
        "Chụp studio một người đàn ông mặc áo dài nam đỏ, đội khăn xếp đen",
    ),
]


def _format_step2a_input(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, indent=2)


def build_step2a_messages(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": STEP2A_SYSTEM}]
    for shot_input, shot_output in STEP2A_FEWSHOT:
        messages.append({"role": "user", "content": _format_step2a_input(shot_input)})
        messages.append({"role": "assistant", "content": shot_output})
    messages.append({"role": "user", "content": _format_step2a_input(spec)})
    return messages


# =============================================================================
# STEP 2b — kiểm tra prompt so với sub_json
# =============================================================================

STEP2B_SYSTEM = r"""Bạn là bộ kiểm duyệt dữ liệu huấn luyện. Bạn nhận:
  1. Một danh sách MỆNH ĐỀ mà câu prompt được phép nói (và phải nói).
  2. Một câu prompt do người khác viết.

Hãy kiểm tra hai điều:
  A. THIẾU  — có mệnh đề nào trong danh sách mà prompt hoàn toàn không nhắc tới không?
             (Diễn đạt khác đi nhưng cùng nghĩa thì VẪN TÍNH LÀ CÓ nhắc tới.)
  B. THÊM   — prompt có nói ra thông tin cụ thể nào KHÔNG hề có trong danh sách không?
             (Từ nối, cách hành văn, lời dẫn kiểu "mình muốn một tấm ảnh" KHÔNG tính là thêm.
              Chỉ tính khi thêm vật thể, màu sắc, số lượng, hành động, địa điểm mới.)

Ngoài ra đánh dấu "robotic": true nếu prompt đọc như một bản dịch máy của dữ liệu có cấu trúc —
liệt kê máy móc từng vật thể, hoặc dùng ngôn ngữ kỹ thuật về khung hình
("nằm ở góc trên bên trái", "chiếm phần dưới khung hình", "ở tiền cảnh").

Chỉ trả về đúng một JSON object, không giải thích:
{
  "missing": ["mệnh đề bị thiếu"],
  "extra": ["thông tin bị thêm"],
  "robotic": false,
  "verdict": "pass"
}
verdict là "pass" khi missing rỗng, extra rỗng và robotic bằng false. Ngược lại là "fail"."""


def build_step2b_messages(facts: List[str], prompt_text: str) -> List[Dict[str, str]]:
    payload = {"menh_de_cho_phep": facts, "prompt": prompt_text}
    return [
        {"role": "system", "content": STEP2B_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
