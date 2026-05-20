#!/usr/bin/env python3
"""Làm giàu dữ liệu sản phẩm từ CSV thô thành JSON có cấu trúc.

Input:
- data/dataset/data.csv

Output:
- data/dataset/data_enriched.json

Script này chuẩn hóa text, parse các trường như giá/thành phần/hướng dẫn dùng,
và sinh thêm các trường suy luận bằng rule để RAG có context giàu hơn.
"""
import csv
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "dataset" / "data.csv"
OUTPUT_JSON = ROOT / "data" / "dataset" / "data_enriched.json"


LOAI_LABELS = {
    "phan_bon": "Phân bón",
    "thuoc_tru_benh": "Thuốc trừ bệnh",
    "thuoc_tru_sau": "Thuốc trừ sâu",
    "thuoc_tru_co": "Thuốc trừ cỏ",
    "thuoc_kich_thich_sinh_truong": "Thuốc kích thích sinh trưởng",
}

STAGE_RULES = [
    ("cây con|sau trồng|bén rễ|bón lót|gieo|mạ", "Cây con"),
    ("sinh trưởng|thân lá|đẻ nhánh|phát triển mạnh", "Giai đoạn sinh trưởng mạnh"),
    ("trước ra hoa|phân hóa mầm hoa|ra hoa", "Trước ra hoa"),
    ("sau đậu trái|đậu trái", "Sau đậu trái"),
    ("nuôi trái|trái lớn|lớn trái|chắc hạt|làm đòng|nuôi củ", "Nuôi trái, củ hoặc hạt"),
    ("sau thu hoạch|phục hồi", "Sau thu hoạch, phục hồi cây"),
    ("trước thu hoạch", "Trước thu hoạch"),
    ("sau mưa|ngập úng|hạn|mặn|thời tiết", "Sau điều kiện thời tiết bất lợi"),
]

SOIL_ENV_RULES = [
    ("humic|fulvic|hữu cơ|đạm cá|rong biển", "Đất nghèo hữu cơ"),
    ("đất phèn|đất chua|\\bpH\\b|thiếu lân|lân khó hấp thu", "Đất phèn hoặc đất chua"),
    ("kẽm|zn|bo|boron|vi lượng|te|mn|cu|fe|mo", "Đất thiếu vi lượng"),
    ("canxi|cao|calcium|bo", "Đất canh tác lâu năm, dễ thiếu Canxi-Bo"),
    ("kali|k2o|kcl|clo", "Đất cần bổ sung Kali cho giai đoạn nuôi trái"),
    ("magie|mg|sulphate|sulfate|lưu huỳnh", "Đất bạc màu, thiếu Magie hoặc lưu huỳnh"),
    ("nấm|mốc|sương mai|thán thư|đạo ôn|mưa nhiều|độ ẩm cao", "Môi trường ẩm cao, dễ phát sinh nấm bệnh"),
    ("cỏ|tiền nảy mầm|hậu nảy mầm", "Ruộng hoặc vườn có áp lực cỏ dại"),
    ("hạn|mặn|nắng nóng|stress", "Điều kiện hạn, mặn hoặc sốc thời tiết"),
]

FEATURE_RULES = [
    ("nội hấp|lưu dẫn|systemic", "Tác động nội hấp, lưu dẫn trong cây"),
    ("tiếp xúc", "Tác động tiếp xúc"),
    ("vị độc", "Tác động vị độc"),
    ("xông hơi", "Tác động xông hơi"),
    ("ức chế|ngăn|phòng", "Có khả năng phòng ngừa và ức chế phát triển"),
    ("lây lan|bào tử|gió|mưa", "Có nguy cơ lây lan nhanh qua gió hoặc nước mưa"),
    ("vi khuẩn|bacterial|zinc thiazole|kasugamycin|iodine", "Tác động lên bệnh do vi khuẩn"),
    ("nấm|mancozeb|copper|metalaxyl|azoxystrobin|tebuconazole|sulfur", "Tác động lên bệnh do nấm"),
    ("sâu|rầy|bọ|nhện|sùng|bọ trĩ|sâu cuốn lá", "Phù hợp xử lý nhóm côn trùng gây hại"),
    ("cỏ|glyphosate|glufosinate|atrazine|quizalofop|penoxsulam", "Phù hợp quản lý cỏ dại"),
    ("thiếu|dinh dưỡng|npk|kali|lân|đạm|canxi|bo|kẽm|magie", "Bổ sung dinh dưỡng thiếu hụt cho cây"),
    ("stress|phục hồi|amino|rong biển|brassinolide|humic", "Giúp cây phục hồi sau stress sinh lý"),
]

MIX_RULES = [
    ("phan_bon", "vi lượng|zn|bo|te|canxi|amino|rong biển", ["Chất bám dính sinh học", "Phân bón lá Amino Acid"]),
    ("phan_bon", "humic|hữu cơ|rễ|lân", ["NPK cân đối", "Trichoderma hoặc chế phẩm cải tạo đất"]),
    ("phan_bon", "kali|nuôi trái|đậu trái", ["Canxi Bo", "Phân bón lá vi lượng"]),
    ("thuoc_tru_benh", "nấm|mốc|sương mai|thán thư|đạo ôn", ["Chất bám dính", "Phân bón lá phục hồi sau bệnh"]),
    ("thuoc_tru_benh", "vi khuẩn|thối|cháy bìa", ["Đồng sinh học", "Amino Acid phục hồi cây"]),
    ("thuoc_tru_sau", "sâu|rầy|bọ|nhện", ["Chất bám dính", "Phân bón lá giúp cây phục hồi"]),
    ("thuoc_tru_co", "cỏ", ["Chất bám dính", "Phân hữu cơ phục hồi đất sau xử lý cỏ"]),
    ("thuoc_kich_thich_sinh_truong", "ra hoa|đậu trái|phục hồi|stress", ["Phân bón lá vi lượng", "Amino Acid hoặc rong biển"]),
]


# Tách một ô CSV thành danh sách text sạch theo các ký tự phân cách.
def split_list(value, separators=r";|,|\n"):
    """Tách một ô CSV thành danh sách sạch, ví dụ công dụng hoặc triệu chứng."""
    value = (value or "").strip()
    if not value:
        return []
    pieces = re.split(separators, value)
    return [clean_text(piece) for piece in pieces if clean_text(piece)]


# Chuẩn hóa khoảng trắng và loại bỏ khoảng trắng thừa ở đầu/cuối.
def clean_text(value):
    """Chuẩn hóa khoảng trắng để text nhất quán hơn."""
    return re.sub(r"\s+", " ", (value or "").strip())


# Parse trường thành phần thành dict nếu có dạng "tên: giá trị".
def parse_thanh_phan(value):
    """Chuyển chuỗi thành phần thành dict nếu có dạng key: value."""
    result = {}
    for part in split_list(value, separators=r";|\n"):
        if ":" in part:
            key, val = part.split(":", 1)
            key = normalize_key(key)
            result[key] = clean_text(val)
        else:
            result[normalize_key(part)] = part
    return result or clean_text(value)


# Chuẩn hóa tên khóa tiếng Việt thành dạng không dấu, dùng được trong JSON.
def normalize_key(value):
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    replacements = {
        "đ": "d",
        " ": "_",
        "-": "_",
        "/": "_",
        "%": "",
        ".": "",
        "(": "",
        ")": "",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "thanh_phan"


# Parse giá từ chuỗi tự nhiên thành dict gồm từ/đến/đơn vị.
def parse_price(value):
    """Parse giá tham khảo thành khoảng từ/đến và đơn vị."""
    numbers = [int(num.replace(".", "")) for num in re.findall(r"\d[\d.]*", value or "")]
    unit = "VND"
    lower = (value or "").lower()
    if "bao" in lower:
        unit = "VND/bao"
    elif "chai" in lower:
        unit = "VND/chai"
    elif "kg" in lower:
        unit = "VND/kg"
    if len(numbers) >= 2:
        return {"tu": numbers[0], "den": numbers[1], "don_vi": unit}
    if len(numbers) == 1:
        return {"tu": numbers[0], "den": numbers[0], "don_vi": unit}
    return {"mo_ta": clean_text(value), "don_vi": unit}


# Rút trích cách dùng, liều lượng và lượng nước từ hướng dẫn sử dụng.
def parse_usage(value):
    """Rút trích cách dùng, liều lượng và lượng nước từ mô tả hướng dẫn."""
    text = clean_text(value)
    usage = {"cach_dung": text}
    dose = re.search(r"(\d+(?:[,.]\d+)?\s*(?:g|kg|ml|lít|lit|cc)(?:\s*/\s*\d+\s*lít)?)", text, re.I)
    water = re.search(r"(\d+(?:[,.]\d+)?(?:-\d+(?:[,.]\d+)?)?\s*lít)", text, re.I)
    if dose:
        usage["lieu_luong"] = dose.group(1)
    if water:
        usage["luong_nuoc"] = water.group(1)
    return usage


# Ghép danh sách triệu chứng thành khóa mô tả ngắn.
def symptoms_key(symptoms):
    return "; ".join(symptoms) if symptoms else "Chưa xác định triệu chứng"


# Tạo rule_key kết hợp loại sản phẩm và triệu chứng để định danh tình huống.
def rule_key(row, symptoms):
    return f"{row['loai']} | {symptoms_key(symptoms)}"


# Dò text bằng các rule regex để suy ra nhãn phù hợp.
def collect_by_rules(text, rules, fallback):
    """Dò text bằng rule regex để suy ra nhãn phù hợp."""
    found = []
    lower = text.lower()
    for pattern, label in rules:
        if re.search(pattern, lower, re.I) and label not in found:
            found.append(label)
    return found or fallback


# Gợi ý sản phẩm/chế phẩm phối hợp dựa trên loại và nội dung mô tả.
def suggest_mixes(row, text):
    """Gợi ý phối hợp thêm sản phẩm/chế phẩm dựa trên loại và nội dung."""
    found = []
    for loai, pattern, suggestions in MIX_RULES:
        if row["loai"] == loai and re.search(pattern, text, re.I):
            for suggestion in suggestions:
                if suggestion not in found:
                    found.append(suggestion)
    if found:
        return found
    if row["loai"] == "phan_bon":
        return ["Chất bám dính sinh học", "Phân hữu cơ cải tạo đất"]
    if row["loai"].startswith("thuoc_tru"):
        return ["Chất bám dính", "Phân bón lá phục hồi cây"]
    return ["Phân bón lá vi lượng", "Amino Acid hoặc rong biển"]


# Sinh câu giải thích ngắn theo kiểu chuyên gia cho sản phẩm.
def expert_reason(row):
    """Sinh lý do chuyên gia ngắn gọn để LLM có căn cứ giải thích."""
    product = row["ten_san_pham"]
    benefit = clean_text(row["cong_dung"]).rstrip(".")
    cause = clean_text(row["nguyen_nhan"]).rstrip(".")
    timing = clean_text(row["thoi_diem_xu_ly"]).rstrip(".")
    if cause:
        return (
            f"{product} phù hợp vì sản phẩm tập trung xử lý nguyên nhân: {cause}. "
            f"Nhờ đó, cây được hỗ trợ {benefit.lower()}, đặc biệt nên dùng ở thời điểm {timing.lower()}."
        )
    return (
        f"{product} giúp {benefit.lower()}. "
        f"Sản phẩm nên được dùng ở thời điểm {timing.lower()} để đạt hiệu quả ổn định hơn."
    )


# Làm giàu một dòng CSV thô thành record metadata có cấu trúc.
def enrich_row(row):
    """Làm giàu một dòng CSV thành một record metadata hoàn chỉnh."""
    symptoms = split_list(row["trieu_chung"], separators=r";|\n")
    timing = split_list(row["thoi_diem_xu_ly"], separators=r";|,|\n")
    text = " ".join(clean_text(row.get(field, "")) for field in row)
    causes = split_list(row["nguyen_nhan"])
    features = collect_by_rules(
        text,
        FEATURE_RULES,
        causes[:2] or ["Đặc tính được suy luận từ công dụng và nguyên nhân sử dụng"],
    )
    stages = collect_by_rules(
        f"{row['thoi_diem_xu_ly']} {row['cong_dung']} {row['trieu_chung']}",
        STAGE_RULES,
        timing or ["Theo thời điểm xử lý khuyến nghị trên nhãn"],
    )
    soil_env = collect_by_rules(
        f"{row['thanh_phan']} {row['nguyen_nhan']} {row['trieu_chung']}",
        SOIL_ENV_RULES,
        ["Điều kiện canh tác thông thường"],
    )

    return {
        "product_id": row["product_id"],
        "ten_san_pham": row["ten_san_pham"],
        "loai": row["loai"],
        "thanh_phan": parse_thanh_phan(row["thanh_phan"]),
        "quy_cach": row["quy_cach"],
        "cong_dung": split_list(row["cong_dung"], separators=r";|,|\n"),
        "trieu_chung": symptoms,
        "nguyen_nhan": causes,
        "doi_tuong_cay_trong": split_list(row["doi_tuong_cay_trong"], separators=r";|,|\n"),
        "thoi_diem_xu_ly": timing,
        "huong_dan_su_dung": parse_usage(row["huong_dan_su_dung"]),
        "an_toan_su_dung": split_list(row["an_toan_su_dung"], separators=r";|\n"),
        "gia": parse_price(row["gia"]),
        "url_img": row["url_img"],
        "rule_key": rule_key(row, symptoms),
        "dac_tinh_benh": features,
        "giai_doan_phu_hop": stages,
        "loai_dat_moi_truong": soil_env,
        "goi_y_phoi_hop": suggest_mixes(row, text),
        "ly_do_chuyen_gia": expert_reason(row),
    }


# Hàm chạy từ terminal: đọc CSV, làm giàu toàn bộ record và ghi JSON.
def main():
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    enriched = [enrich_row(row) for row in rows]
    seen = set()
    duplicate_rule_keys = []
    for item in enriched:
        if item["rule_key"] in seen:
            duplicate_rule_keys.append(item["rule_key"])
        seen.add(item["rule_key"])

    OUTPUT_JSON.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(enriched)} records to {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Duplicate rule_key count: {len(duplicate_rule_keys)}")


if __name__ == "__main__":
    main()
