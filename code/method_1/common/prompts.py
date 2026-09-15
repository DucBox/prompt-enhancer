"""Toàn bộ system prompt + few-shot của pipeline.

Gom một chỗ để sửa prompt không phải đụng vào logic từng step.
Ba prompt tương ứng ba lần gọi model:
    STEP1A  phân rã target_json thành nhóm khái niệm + mệnh đề nguyên tử
    STEP2A  viết user prompt từ sub_json (giọng người dùng thật)
    STEP2B  kiểm tra prompt có thêm/thiếu so với sub_json không
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# =============================================================================
# STEP 1a — gom nhóm khái niệm & phân rã mệnh đề
# =============================================================================

STEP1A_SYSTEM = r"""Bạn là công cụ phân tích mô tả ảnh. Đầu vào gồm GỢI Ý CHỦ ĐỀ (goi_y_chu_de, lấy từ tên
thư mục chứa ảnh) và một JSON mô tả chi tiết một bức ảnh.
Nhiệm vụ của bạn gồm NĂM phần. Chỉ trả về đúng một JSON object, không giải thích gì thêm.

Kết quả sẽ được một bước sau dùng để CHỌN LỌC thông tin cho các yêu cầu ngắn / vừa / dài.
Vì vậy ở bước này:
- TUYỆT ĐỐI KHÔNG bỏ bớt thông tin. Phải tách ĐẦY ĐỦ mọi ý có trong mô tả gốc — chọn giữ
  hay bỏ là việc của bước sau.
- Phải XẾP HẠNG thật chuẩn, vì bước sau chọn theo thứ hạng.

=== LUẬT CHUNG CHO MỌI MỆNH ĐỀ ===
- Mỗi mệnh đề chỉ nói MỘT ý, viết bằng TIẾNG VIỆT, ngắn (2-6 từ), đọc riêng vẫn hiểu
  (không cụt câu, không lơ lửng giới từ).
- Giữ NGUYÊN VĂN tên riêng và thuật ngữ văn hoá Việt.
- Không bịa thêm ý không có trong mô tả gốc. Không gộp hai ý vào một mệnh đề.
- Tách theo NGHĨA, không theo dấu câu: "eye-level wide shot" là HAI ý dù không có dấu phẩy;
  một vế có "with" cũng có thể chứa nhiều ý.
- "rank" đánh từ 0, xếp theo độ QUAN TRỌNG GIẢM DẦN, TÍNH RIÊNG TRONG TỪNG TRƯỜNG.
  Hãy dùng TOÀN BỘ JSON (high_level_description, chủ thể chính, các element khác) làm ngữ cảnh
  để xếp hạng: thông tin làm nên nét RIÊNG của bức ảnh này — thứ người dùng nhiều khả năng
  nhắc tới nhất — đứng trước; thông tin chung chung mà ảnh nào cũng có đứng sau.

=== PHẦN 0: CHỦ ĐỀ CHÍNH (chu_de_chinh, khop_goi_y) ===
Đọc high_level_description và trả lời: bức ảnh này VỀ CÁI GÌ? Viết 1-3 mệnh đề tiếng Việt
(tổng không quá 10 từ) — thứ mà người dùng BẮT BUỘC phải nói khi đặt hàng đúng bức ảnh này,
kể cả ở yêu cầu ngắn nhất. Mọi yêu cầu sinh ra sau này đều phải truyền tải đúng chủ đề này.
- Bám theo high_level_description, KHÔNG bám theo element nào được tả nhiều nhất hay nhiều
  người/vật nhất. Ví dụ: mô tả "a group of four people gathered around a pottery wheel,
  shaping wet clay" thì chủ đề là ["nhóm người làm gốm", "quanh bàn xoay gốm"], KHÔNG phải
  ["hai người phụ nữ", "một người đàn ông"].
- Chủ đề là hoạt động / sự kiện / cả khung cảnh thì nêu đúng hoạt động / khung cảnh đó
  (làm gốm, đua ghe, chợ đường phố...), không thay bằng danh sách người/vật tham gia.
- Chỉ lấy ý CỐT LÕI. Không đưa màu sắc, chất liệu, ánh sáng, góc máy vào trừ khi chính nó làm
  nên chủ đề (vd "tranh khắc gỗ").
- goi_y_chu_de CHỈ ĐỂ THAM KHẢO cách gọi tên, KHÔNG bắt buộc phải xuất hiện:
    * Ảnh thật sự thể hiện chủ đề gợi ý (kể cả khi mô tả gốc viết tiếng Anh hoặc gọi khác đi,
      vd "ceramic teapot" với gợi ý "ấm tích") → gọi tên chủ đề bằng "thuat_ngu" hoặc một từ
      trong "dong_nghia" hợp với ảnh nhất, và đặt "khop_goi_y": true.
    * Mô tả gốc gọi CỤ THỂ hơn hoặc khác gợi ý mà vẫn đúng (gợi ý "Thành phố Hồ Chí Minh" nhưng
      ảnh là "chợ đường phố Sài Gòn"; ảnh là một công trình cụ thể) → gọi theo mô tả gốc, vẫn
      giữ tên địa danh nếu mô tả có, và đặt "khop_goi_y": true.
    * Ảnh KHÔNG thể hiện chủ đề gợi ý (tên thư mục chỉ là từ khoá tìm kiếm) → bỏ qua gợi ý,
      đặt "khop_goi_y": false. TUYỆT ĐỐI không ép thuật ngữ vào khi ảnh không có.
- "rank" theo LUẬT CHUNG; mệnh đề rank 0 là cái cốt lõi nhất.

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

=== PHẦN 2: PHÂN RÃ ELEMENT (facts) ===
Với MỖI element, tách "desc" thành các mệnh đề theo LUẬT CHUNG.
- Mệnh đề rank 0 LUÔN là danh từ chủ thể trần trụi.
- Các mệnh đề sau: màu sắc, chất liệu, công dụng thường quan trọng hơn vị trí trong khung hình.
- Nếu element có trường "text" (chữ xuất hiện trong ảnh), phải có một mệnh đề chứa NGUYÊN VĂN chữ đó.
- "kind" thuộc: subject | color | material | attribute | action | position | count

=== PHẦN 3: PHÂN RÃ BỐI CẢNH (background_facts) ===
Tách compositional_deconstruction.background thành các mệnh đề theo LUẬT CHUNG, dùng cùng bộ
"kind" như PHẦN 2.
- Mệnh đề rank 0 là không gian hoặc bề mặt chính của bối cảnh (ví dụ "mặt bàn gỗ", "bầu trời", "mặt sông").
- Bối cảnh nhắc tới vật cụ thể nào (cái bát, giá treo đồ, hàng cây...) thì vật đó là một mệnh đề
  riêng; các đặc điểm của nó (màu, chất liệu, vị trí, độ mờ...) là các mệnh đề riêng khác.

=== PHẦN 4: PHÂN RÃ PHONG CÁCH (style_facts) ===
Tách từng trường của style_description thành các mệnh đề theo LUẬT CHUNG (không có "kind"):
- "photo"      (nếu có): góc máy, cỡ cảnh, lấy nét, ống kính... — mỗi ý một mệnh đề.
- "art_style"  (nếu có): kỹ thuật vẽ hoặc dựng hình, nét, cách tô màu, chất liệu nền... — mỗi ý một mệnh đề.
- "lighting"  : nguồn sáng, cường độ, hướng, màu ánh sáng, bóng đổ... — mỗi ý một mệnh đề.
- "aesthetics": mỗi tag là một mệnh đề, dịch sang tiếng Việt.
KHÔNG phân rã "medium" và "color_palette". Chỉ tạo khoá cho trường THẬT SỰ có trong đầu vào
("photo" và "art_style" không bao giờ cùng xuất hiện).

=== ĐỊNH DẠNG ĐẦU RA ===
{
  "chu_de_chinh": [{"rank": 0, "text": "..."}],
  "khop_goi_y": true,
  "concept_groups": [
    {"name": "...", "member_ids": [0,1], "cardinality": "exact:2", "cultural": false}
  ],
  "facts": {
    "0": [
      {"rank": 0, "kind": "subject", "text": "..."},
      {"rank": 1, "kind": "color", "text": "..."}
    ]
  },
  "background_facts": [
    {"rank": 0, "kind": "subject", "text": "..."}
  ],
  "style_facts": {
    "photo": [{"rank": 0, "text": "..."}],
    "lighting": [{"rank": 0, "text": "..."}],
    "aesthetics": [{"rank": 0, "text": "..."}]
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
        "lighting": "bright daylight with soft shadows",
        "photo": "top-down aerial, wide shot, deep focus",
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

STEP1A_FEWSHOT_HINT = {
    "ten_thu_muc": "cho noi cai rang",
    "thuat_ngu": "chợ nổi Cái Răng",
    "dong_nghia": ["Cái Răng"],
}

STEP1A_FEWSHOT_OUTPUT = {
    "chu_de_chinh": [{"rank": 0, "text": "chợ nổi Cái Răng"}],
    "khop_goi_y": True,
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
    "background_facts": [
        {"rank": 0, "kind": "subject", "text": "mặt sông"},
        {"rank": 1, "kind": "attribute", "text": "rộng"},
        {"rank": 2, "kind": "color", "text": "nước nâu đục"},
        {"rank": 3, "kind": "position", "text": "lấp đầy khung hình"},
    ],
    "style_facts": {
        "photo": [
            {"rank": 0, "text": "chụp từ trên không"},
            {"rank": 1, "text": "nhìn thẳng từ trên xuống"},
            {"rank": 2, "text": "toàn cảnh"},
            {"rank": 3, "text": "lấy nét sâu"},
        ],
        "lighting": [
            {"rank": 0, "text": "ánh sáng ban ngày"},
            {"rank": 1, "text": "trời sáng rõ"},
            {"rank": 2, "text": "bóng đổ mềm"},
        ],
        "aesthetics": [
            {"rank": 0, "text": "sống động"},
            {"rank": 1, "text": "phong cách phóng sự"},
        ],
    },
}


def _format_step1a_input(target_json: Dict[str, Any], hint: Optional[Dict[str, Any]]) -> str:
    dumps = lambda o: json.dumps(o, ensure_ascii=False, indent=2)  # noqa: E731
    return "GỢI Ý CHỦ ĐỀ (goi_y_chu_de — chỉ tham khảo):\n{}\n\nMÔ TẢ ẢNH:\n{}".format(
        dumps(hint or {}), dumps(target_json))


def build_step1a_messages(
    target_json: Dict[str, Any], hint: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": STEP1A_SYSTEM},
        {"role": "user", "content": _format_step1a_input(STEP1A_FEWSHOT_INPUT, STEP1A_FEWSHOT_HINT)},
        {"role": "assistant", "content": json.dumps(STEP1A_FEWSHOT_OUTPUT, ensure_ascii=False, indent=2)},
        {"role": "user", "content": _format_step1a_input(target_json, hint)},
    ]


# =============================================================================
# STEP 1b — LLM chọn lọc thông tin cho short / medium / long
# =============================================================================

from common import subjson  # noqa: E402  (chỉ để dựng few-shot; subjson không import prompts)

STEP1B_SYSTEM = r"""Bạn là công cụ CHỌN LỌC thông tin mô tả ảnh cho ba mức yêu cầu của người dùng:
ngắn (short), vừa (medium), dài (long).

Đầu vào là bản phân rã của MỘT bức ảnh. Mọi mệnh đề đã được viết bằng tiếng Việt và xếp theo
độ quan trọng GIẢM DẦN trong từng danh sách:
- "high_level_description": câu tóm tắt ảnh — chỉ để hiểu ngữ cảnh, không chọn.
- "chu_de_chinh": các mệnh đề nói bức ảnh VỀ CÁI GÌ. Code TỰ ĐƯA chúng vào cả ba mức — KHÔNG
  chọn lại, không tính vào số ý hay số từ của mức nào.
- "medium": loại ảnh (photograph, illustration, 3d_render...).
- "groups": các nhóm khái niệm, nhóm đứng đầu là quan trọng nhất. Mỗi nhóm có "name",
  "cultural" (mang bản sắc Việt Nam), "cardinality", có thể có "so_luong" (vd "số lượng: 3"),
  và "elements" = danh sách mệnh đề của từng cá thể (mệnh đề đầu tiên là danh từ chủ thể).
- "background": mệnh đề về bối cảnh.
- "style": mệnh đề về góc máy hoặc kỹ thuật vẽ ("photo" / "art_style"), ánh sáng ("lighting"),
  thẩm mỹ ("aesthetics").

Nhiệm vụ: với MỖI mức, chọn ra đúng những gì một người dùng ở mức đó sẽ nói khi đặt hàng
bức ảnh này.

=== ĐỊNH NGHĨA BA MỨC ===
Mức chi tiết được quyết định bởi ĐỘ PHỦ (nói tới bao nhiêu phần của ảnh), ĐỘ SÂU (mỗi phần tả
kỹ tới đâu) và LOẠI THÔNG TIN — KHÔNG phải số từ.

SHORT — người dùng chỉ nói thứ họ quan tâm nhất, bỏ qua phần còn lại của ảnh.
- Chủ đề chính đã có sẵn. Được chọn thêm nhóm chủ thể chính (cùng danh từ chủ thể của nó) nếu
  nó làm rõ chủ đề; chủ đề chính đã đủ thì "groups" để rỗng.
- Thêm TỐI ĐA 2 ý, chọn trong số:
    * thuộc tính nổi bật nhất của chủ thể chính;
    * HOẶC một bối cảnh / địa điểm CÓ BẢN SẮC (Hồ Tây, phố cổ Hội An, cánh đồng lúa, bãi
      biển...), lấy từ "background" HOẶC từ một nhóm văn hoá — không lấy cả hai nguồn.
  Mỗi thuộc tính, "so_luong", ý bối cảnh, nhóm thêm ngoài chủ thể chính, hay đặt "medium"
  là true đều tính là MỘT ý.
- KHÔNG lấy bối cảnh chung chung (mặt bàn gỗ, phông nền trắng, bức tường...).
- KHÔNG lấy "style".
- Tổng số từ của mọi mệnh đề (kể cả tên nhóm văn hoá, không tính chủ đề chính) KHÔNG quá 16.

MEDIUM — người dùng tả những thứ chính của ảnh nhưng chưa đi vào chi tiết vụn.
- Chủ thể chính với 2-4 thuộc tính nổi bật.
- 1-3 nhóm phụ quan trọng, mỗi nhóm 1-2 thuộc tính.
- 1-2 ý bối cảnh chính.
- Tối đa 1 ý "style", CHỈ KHI nó thật sự đáng chú ý với bức ảnh này (chụp từ trên cao,
  flat lay, tranh khắc gỗ...). Không lấy thứ ảnh nào cũng có (ngang tầm mắt, lấy nét sâu,
  ánh sáng ban ngày...).
- Tổng số từ (không tính chủ đề chính) KHÔNG quá 50.

LONG — người dùng tả gần như toàn bộ bức ảnh.
- Mọi nhóm, với thuộc tính, vị trí, hành động.
- Bối cảnh chi tiết.
- Ánh sáng, góc máy, phong cách.
- Được bỏ mệnh đề trùng lặp hoặc quá vụn vặt, nhưng phần lớn thông tin phải có mặt.

=== LUẬT CHỌN ===
1. Chép NGUYÊN VĂN mệnh đề và tên nhóm từ đầu vào. Không sửa chữ, không gộp, không dịch,
   không bịa.
2. Tên nhóm thường chỉ là nhãn. Muốn nói tới thứ gì thì chọn mệnh đề của nó, kể cả danh từ
   chủ thể. Riêng nhóm "cultural": true, tên nhóm được tính là đã nói.
3. Ưu tiên theo thứ tự xếp hạng, nhưng được bỏ qua mệnh đề hạng cao nếu nó chung chung,
   không đáng nói ở mức đó.
4. LỒNG NHAU: mọi thứ có ở short phải có ở medium; mọi thứ có ở medium phải có ở long.
5. Mọi thứ chọn thêm phải phục vụ "chu_de_chinh". Ưu tiên ý làm rõ chủ đề (hoạt động, địa điểm
   có bản sắc, vật đặc trưng) hơn chi tiết lẻ về từng người/vật; KHÔNG chọn sao cho chủ đề bị
   lấn át (ảnh làm gốm mà mức short chỉ liệt kê "người phụ nữ", "người đàn ông" là SAI).
6. "so_luong" chép vào "facts" của nhóm khi con số là thông tin đáng nói. KHÔNG chọn khi tên
   nhóm đã tự thể hiện số lượng (đôi, cặp, bộ...).
7. "medium": đặt true khi loại ảnh là thông tin đáng nói (tranh minh hoạ, ảnh dựng 3D...);
   ảnh chụp thông thường thì thường không cần.
8. Đừng chọn theo một khuôn cố định cho mọi ảnh: số lượng ý thay đổi theo độ nổi bật thực tế
   của từng bức ảnh, trong giới hạn của mỗi mức.

=== ĐỊNH DẠNG ĐẦU RA ===
Chỉ trả về đúng một JSON object, không giải thích:
{
  "short":  {"groups": [{"name": "...", "facts": ["..."]}], "background": [], "style": {}, "medium": false},
  "medium": {"groups": [...], "background": [...], "style": {...}, "medium": false},
  "long":   {"groups": [...], "background": [...], "style": {...}, "medium": false}
}
- "style" là object, khoá thuộc "photo" / "art_style" / "lighting" / "aesthetics" (chỉ khoá có
  trong đầu vào), mỗi khoá là danh sách mệnh đề.
- "facts" của mỗi nhóm là danh sách mệnh đề chọn từ "elements" của nhóm (và "so_luong" nếu chọn)."""


STEP1B_FEWSHOT_INPUT = subjson.build_selection_input(STEP1A_FEWSHOT_INPUT, STEP1A_FEWSHOT_OUTPUT)

# Nhóm "chợ nổi Cái Răng" trùng chủ đề chính nên không chọn lại -- code đã tự đưa vào mọi mức.
STEP1B_FEWSHOT_OUTPUT = {
    "short": {
        "groups": [
            {"name": "những chiếc thuyền gỗ", "facts": ["chiếc thuyền", "bằng gỗ"]},
        ],
        "background": [],
        "style": {},
        "medium": False,
    },
    "medium": {
        "groups": [
            {"name": "những chiếc thuyền gỗ",
             "facts": ["chiếc thuyền", "số lượng: 2", "bằng gỗ", "màu tím", "là quán ăn nổi", "màu xanh"]},
            {"name": "người bán hàng đội nón lá", "facts": ["đội nón lá", "đang nấu đồ ăn"]},
        ],
        "background": ["mặt sông", "nước nâu đục"],
        "style": {"photo": ["chụp từ trên không"]},
        "medium": False,
    },
    "long": {
        "groups": [
            {"name": "những chiếc thuyền gỗ",
             "facts": ["chiếc thuyền", "số lượng: 2", "bằng gỗ", "màu tím", "là quán ăn nổi", "dài",
                       "nằm ngang giữa khung hình", "có khách trên thuyền", "màu xanh",
                       "chở khách ngồi thành hàng"]},
            {"name": "người bán hàng đội nón lá",
             "facts": ["những người bán hàng", "đội nón lá", "đang nấu đồ ăn"]},
            {"name": "cảnh nền", "facts": ["cái xô nhựa", "màu đỏ", "trên boong thuyền"]},
        ],
        "background": ["mặt sông", "rộng", "nước nâu đục", "lấp đầy khung hình"],
        "style": {
            "photo": ["chụp từ trên không", "nhìn thẳng từ trên xuống", "toàn cảnh"],
            "lighting": ["ánh sáng ban ngày", "trời sáng rõ"],
            "aesthetics": ["sống động"],
        },
        "medium": False,
    },
}


def build_step1b_messages(sel_input: Dict[str, Any]) -> List[Dict[str, str]]:
    dumps = lambda o: json.dumps(o, ensure_ascii=False, indent=2)  # noqa: E731
    return [
        {"role": "system", "content": STEP1B_SYSTEM},
        {"role": "user", "content": dumps(STEP1B_FEWSHOT_INPUT)},
        {"role": "assistant", "content": dumps(STEP1B_FEWSHOT_OUTPUT)},
        {"role": "user", "content": dumps(sel_input)},
    ]


def build_step1b_fix_message(errors: List[str]) -> str:
    return ("Lựa chọn vừa rồi CHƯA hợp lệ:\n- " + "\n- ".join(errors)
            + "\nHãy trả lại JSON đầy đủ cho cả ba mức, sửa đúng các lỗi trên và giữ nguyên "
              "mọi luật chọn.")


# =============================================================================
# STEP 2a — viết user prompt từ sub_json
# =============================================================================

STEP2A_SYSTEM = r"""Bạn đang đóng vai NGƯỜI DÙNG muốn nhờ AI tạo ra một tấm ảnh.
Bạn được cho một bản mô tả có cấu trúc. Hãy viết ra CÂU YÊU CẦU của bạn.
Chỉ trả về đúng câu yêu cầu đó, không thêm lời dẫn, không giải thích, không đặt trong ngoặc kép.

Bản mô tả gồm: "chu_de_chinh" (bức ảnh VỀ CÁI GÌ), "groups" (mỗi nhóm là các mệnh đề về một thứ
trong ảnh), "boi_canh" (bối cảnh) và "phong_cach" (góc chụp, ánh sáng, phong cách, loại ảnh) nếu có.

BẮT BUỘC
- Viết như người ĐANG ĐẶT HÀNG một bức ảnh chưa tồn tại,
  KHÔNG phải người đang mô tả một bức ảnh có sẵn trước mặt.
- Chỉ nói những gì có trong bản mô tả. Cấm thêm bất kỳ thông tin mới nào.
- "chu_de_chinh" là ý chính: đọc câu yêu cầu phải hiểu ngay bức ảnh về cái gì (thường nói ngay từ
  đầu). Các mệnh đề khác chỉ bổ sung cho chủ đề, KHÔNG được lấn át nó — ảnh "nhóm người làm gốm"
  mà câu yêu cầu chỉ nói "hai phụ nữ và một người đàn ông" là SAI.
- PHẢI nhắc đến ĐẦY ĐỦ mọi mệnh đề được cho — trong "chu_de_chinh", "groups", "boi_canh" và
  "phong_cach".
  Được phép diễn đạt lại tự nhiên hơn, gộp chung với ý khác, KHÔNG được tự ý bỏ sót bất kỳ
  mệnh đề nào chỉ vì thấy không quan trọng — thiếu một mệnh đề cũng bị coi là lỗi giống hệt
  như thêm bịa.
- Giữ NGUYÊN VĂN thuật ngữ tiếng Việt (áo dài, khăn xếp, bánh tét, chợ nổi Cái Răng, nón lá...).
- Mệnh đề "số lượng: N" thì phải nói đúng số đó.
  Nhóm có "so_nhieu": true (không kèm số lượng) thì nói mơ hồ ("mấy", "vài", "nhiều"),
  KHÔNG được tự chế ra con số. Nhóm không có hai thứ này là một cá thể.

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
# Bản mô tả đúng dạng step2a_verbalize.build_spec() dựng ra. Câu trả lời nhắc ĐỦ mọi mệnh đề,
# không thêm gì (kể cả vị trí), và viết đúng chính tả.
STEP2A_FEWSHOT = [
    (
        {
            "detail_level": "short",
            "length_hint": "8-20 từ",
            "persona": {"vai": "user phổ thông", "giọng": "mô tả trung tính",
                        "ngôn ngữ": "tiếng Việt"},
            "chu_de_chinh": ["chợ nổi Cái Răng"],
            "groups": [
                {"facts": ["chiếc thuyền", "bằng gỗ"], "so_nhieu": True},
            ],
        },
        "Ảnh chợ nổi Cái Răng với mấy chiếc thuyền gỗ",
    ),
    (
        {
            "detail_level": "medium",
            "length_hint": "30-60 từ",
            "persona": {"vai": "designer", "giọng": "ra lệnh",
                        "ngôn ngữ": "tiếng Việt"},
            "chu_de_chinh": ["chợ nổi Cái Răng"],
            "groups": [
                {"facts": ["chiếc thuyền", "số lượng: 3", "bằng gỗ", "một chiếc màu tím",
                           "là quán ăn nổi"]},
                {"facts": ["những người bán hàng", "đội nón lá", "đang nấu đồ ăn"], "so_nhieu": True},
            ],
            "boi_canh": ["mặt sông", "nước đục"],
            "phong_cach": ["chụp từ flycam"],
        },
        "Chụp từ flycam chợ nổi Cái Răng trên mặt sông nước đục, ba chiếc thuyền gỗ, trong đó "
        "một chiếc màu tím làm quán ăn nổi, mấy người bán hàng đội nón lá đang nấu đồ ăn.",
    ),
    (
        {
            "detail_level": "long",
            "length_hint": "100-200 từ",
            "persona": {"vai": "user phổ thông", "giọng": "kể lể lan man",
                        "ngôn ngữ": "tiếng Việt"},
            "chu_de_chinh": ["chợ nổi Cái Răng"],
            "groups": [
                {"facts": ["chiếc thuyền", "màu tím", "bằng gỗ", "dài", "là quán ăn nổi"]},
                {"facts": ["những người bán hàng", "đội nón lá", "đang nấu đồ ăn"], "so_nhieu": True},
                {"facts": ["chiếc thuyền", "chở sọt trái cây"]},
                {"facts": ["chiếc thuyền", "màu xanh", "chở khách ngồi thành hàng"]},
            ],
            "boi_canh": ["mặt sông", "rộng", "màu nâu đục", "chiếm gần hết khung hình"],
            "phong_cach": ["chụp từ trên cao", "ánh sáng ban ngày"],
        },
        "Mình muốn ảnh chụp từ trên cao cảnh chợ nổi Cái Răng dưới ánh sáng ban ngày. Mặt sông "
        "rộng màu nâu đục chiếm gần hết khung hình. Có một chiếc thuyền gỗ dài màu tím làm quán ăn "
        "nổi, mấy người bán hàng đội nón lá đang nấu đồ ăn, thêm một chiếc thuyền chở sọt trái cây "
        "và một chiếc thuyền màu xanh chở khách ngồi thành hàng.",
    ),
    (
        {
            "detail_level": "short",
            "length_hint": "8-20 từ",
            "persona": {"vai": "user phổ thông", "giọng": "ra lệnh",
                        "ngôn ngữ": "tiếng Việt"},
            "chu_de_chinh": ["áo dài nam", "khăn xếp"],
            "groups": [
                {"facts": ["áo dài nam", "màu đỏ"]},
                {"facts": ["khăn xếp"]},
            ],
        },
        "Cho mình ảnh áo dài nam màu đỏ đi kèm khăn xếp",
    ),
    (
        # Chủ đề là HOẠT ĐỘNG: người trong ảnh chỉ là chi tiết bổ sung, không thay được chủ đề.
        {
            "detail_level": "short",
            "length_hint": "8-20 từ",
            "persona": {"vai": "sinh viên làm đồ án", "giọng": "mô tả trung tính",
                        "ngôn ngữ": "tiếng Việt"},
            "chu_de_chinh": ["nhóm người làm gốm", "quanh bàn xoay gốm"],
            "groups": [
                {"facts": ["cô gái nhỏ"]},
            ],
        },
        "Muốn có tấm ảnh một nhóm người đang làm gốm quanh bàn xoay gốm, trong nhóm có một cô gái nhỏ.",
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
  1. Các MỆNH ĐỀ mà câu prompt được phép nói (và phải nói), xếp THEO NHÓM trong
     "menh_de_theo_nhom":
       - "chu_de_chinh": bức ảnh VỀ CÁI GÌ;
       - "groups": mỗi phần tử là các mệnh đề về MỘT thứ trong ảnh (một vật, một người, hay
         một nhóm cùng loại). "so_nhieu": true nghĩa là thứ đó có nhiều cái nhưng không nêu số;
       - "boi_canh": bối cảnh;  "phong_cach": góc chụp, ánh sáng, phong cách, loại ảnh.
     (Dữ liệu cũ có thể chỉ là một danh sách phẳng "menh_de_cho_phep".)
  2. Một câu prompt do người khác viết.

Hãy kiểm tra hai điều:
  A. THIẾU  — có mệnh đề nào trong danh sách mà prompt hoàn toàn không nhắc tới không?
             (Diễn đạt khác đi nhưng cùng nghĩa thì VẪN TÍNH LÀ CÓ nhắc tới. Điều này áp
              dụng CẢ KHI prompt viết bằng ngôn ngữ khác với mệnh đề — ví dụ mệnh đề ghi
              "bright overcast daylight" mà prompt viết "ánh sáng ban ngày dịu, trời âm u"
              thì vẫn tính là ĐÃ nhắc tới, không phải thiếu. Ngôn ngữ không quan trọng,
              chỉ nghĩa mới quan trọng.)
  B. THÊM   — prompt có nói ra thông tin cụ thể nào KHÔNG hề có trong danh sách không?
             (Từ nối, cách hành văn, lời dẫn kiểu "mình muốn một tấm ảnh" KHÔNG tính là thêm.
              Chỉ tính khi thêm vật thể, màu sắc, số lượng, hành động, địa điểm mới.
              Dịch một mệnh đề sang ngôn ngữ khác KHÔNG tính là thêm, dù không giống chữ.)

NGOẠI LỆ DUY NHẤT — thuật ngữ văn hoá Việt Nam: nếu một mệnh đề là tên riêng hoặc khái niệm
mang bản sắc Việt Nam (ví dụ: áo dài, khăn xếp, chợ nổi Cái Răng, cây bẹo, thanh đồng, lễ Hầu
đồng, bánh chưng...), tên đó BẮT BUỘC phải giữ nguyên bằng tiếng Việt. Nếu prompt dịch nó sang
tiếng Anh hoặc thay bằng một khái niệm khác (ví dụ "áo dài" → "Vietnamese dress", "thanh đồng"
→ "shaman") thì tính là THIẾU mệnh đề đó — đây là lỗi cần bắt được. Các mệnh đề còn lại
(màu sắc, ánh sáng, vị trí, hành động, bối cảnh...) không bị ràng buộc ngôn ngữ.

MỆNH ĐỀ THUỘC VỀ NHÓM CỦA NÓ: một mệnh đề trong "groups" chỉ nói về thứ của CHÍNH nhóm đó. Màu,
chất liệu, hành động của nhóm này nhắc cho thứ khác trong prompt thì KHÔNG tính là đã nhắc.

MỆNH ĐỀ SỐ LƯỢNG: "số lượng: N" là số cá thể của CHÍNH NHÓM chứa nó, không phải của nhóm khác.
  - Nhóm có "người phụ nữ" và "số lượng: 3", prompt viết "ba người phụ nữ" hoặc "3 người phụ nữ"
    → ĐÃ nhắc tới.
  - Prompt nêu số khác, hoặc chỉ nói mơ hồ ("mấy", "vài", "nhiều") cho nhóm đó → THIẾU.
  - Nhóm "so_nhieu": true mà prompt nói mơ hồ ("mấy chiếc thuyền") là ĐÚNG; prompt tự đặt ra một
    con số cụ thể cho nhóm đó → tính là THÊM.

MỆNH ĐỀ VỊ TRÍ CÓ LẶP TÊN CHỦ THỂ: một số mệnh đề vị trí lặp lại tên chủ thể/nhóm ở cuối câu,
ví dụ "ở tai phải thanh đồng", "cầm ở tay trái thanh đồng". Một prompt viết tự nhiên KHÔNG lặp
lại tên chủ thể cho từng chi tiết như vậy (chỉ nhắc ngầm qua ngữ cảnh đã thiết lập) —
CHỈ VÌ THIẾU TÊN CHỦ THỂ LẶP LẠI thì KHÔNG được tính là thiếu.
Nhưng phần vị trí CỤ THỂ vẫn phải đúng — trái/phải, tay/tai/cổ, trên/dưới... phải khớp:
  - Mệnh đề "ở tai phải thanh đồng", prompt viết "...bông tai vàng bên tai phải" → ĐÃ nhắc tới.
  - Mệnh đề "ở tai phải thanh đồng", prompt viết "...bông tai vàng bên tai trái" → SAI vị trí,
    vẫn tính là THIẾU (không được bỏ qua chỉ vì đây là mệnh đề vị trí).
  - Mệnh đề "ở tai phải thanh đồng", prompt không nhắc bên nào cả → THIẾU.

Chỉ trả về đúng một JSON object, không giải thích.
Mỗi phần tử trong "missing" phải CHÉP NGUYÊN VĂN một mệnh đề được cho (chỉ chuỗi mệnh đề, không
kèm tên nhóm) — không diễn đạt lại, không gộp, không tóm tắt:
{
  "missing": ["mệnh đề bị thiếu"],
  "extra": ["thông tin bị thêm"],
  "verdict": "pass"
}
verdict là "pass" khi missing rỗng và extra rỗng. Ngược lại là "fail"."""


def build_step2b_messages(
    facts: List[str], prompt_text: str, grouped: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """`grouped` là mệnh đề theo nhóm (step2a_verbalize.spec_content) -- có thì judge chấm theo
    nhóm; dòng cũ không có thì rơi về danh sách phẳng `facts`."""
    if grouped:
        payload: Dict[str, Any] = {"menh_de_theo_nhom": grouped, "prompt": prompt_text}
    else:
        payload = {"menh_de_cho_phep": facts, "prompt": prompt_text}
    return [
        {"role": "system", "content": STEP2B_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


# =============================================================================
# STEP 2b2 — sửa lại prompt bị 2b loại (retry 1 lần), thay vì bỏ trắng
# =============================================================================

STEP2B2_SYSTEM = r"""Bạn nhận một CÂU PROMPT đã bị một bộ kiểm duyệt từ chối, kèm đúng lý do bị từ chối.
Nhiệm vụ: viết lại câu prompt để SỬA ĐÚNG những lỗi đã nêu, giữ nguyên mọi phần còn lại
(văn phong, độ dài, thứ tự, ngôn ngữ) không cần thiết phải đổi.

Bạn được cho:
  - "prompt_cu": câu prompt gốc.
  - "thieu": các mệnh đề bị đánh giá là THIẾU — phải bổ sung chúng vào (diễn đạt tự
    nhiên, không cần chép nguyên văn).
  - "thua": nội dung bị đánh giá là THÊM không có căn cứ — phải bỏ đúng phần đó ra.
  - "checklist": toàn bộ mệnh đề được phép nói, để không vô tình bịa thêm khi sửa.

Quy tắc:
- CHỈ sửa đúng phần bị nêu lỗi. Không viết lại toàn bộ câu nếu không cần thiết.
- Không được thêm nội dung nào ngoài "checklist".
- Giữ nguyên độ dài tương đối, văn phong, ngôn ngữ của câu gốc.
- Giữ NGUYÊN VĂN thuật ngữ tiếng Việt.
- Chỉ trả về đúng câu prompt đã sửa, không thêm lời dẫn, không giải thích, không đặt
  trong ngoặc kép."""


def build_step2b2_messages(
    checklist: List[str], missing: List[str], extra: List[str], old_prompt: str,
) -> List[Dict[str, str]]:
    payload = {
        "prompt_cu": old_prompt,
        "thieu": missing,
        "thua": extra,
        "checklist": checklist,
    }
    return [
        {"role": "system", "content": STEP2B2_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
