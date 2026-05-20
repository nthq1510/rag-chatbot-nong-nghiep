#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = ROOT / "data" / "dataset" / "data_enriched.json"
OUTPUT_DIR = ROOT / "data" / "Thuoc_Metadata"


# Hàm chạy từ terminal: nhóm data_enriched.json thành các file metadata theo loại.
def main():
    records = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in records:
        grouped[record["loai"]].append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for loai, items in sorted(grouped.items()):
        output_path = OUTPUT_DIR / f"{loai}.json"
        output_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(items):>3} records: {output_path.relative_to(ROOT)}")

    summary = {
        "source": str(INPUT_JSON.relative_to(ROOT)),
        "total_records": len(records),
        "groups": {loai: len(items) for loai, items in sorted(grouped.items())},
    }
    (OUTPUT_DIR / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary: {(OUTPUT_DIR / '_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
