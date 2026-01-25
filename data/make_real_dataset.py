import json
from pathlib import Path

# MoonBoard 18x11
COLS = 11

def get_rc(hid):
    """把数字 ID 转换成行列坐标"""
    # 假设 H0 是 A1 (0,0)
    hid = int(hid.replace("H", ""))
    r = hid // COLS
    c = hid % COLS
    return r, c

def make_real_data():
    # 这里是我们精选的 10 条符合物理规则的路线
    # 包含了起步(S)、中间点、完攀(F)的逻辑
    raw_data = [
        {"grade": 3, "holds": ["H15", "H48", "H82", "H114", "H148", "H181"], "start": "H15", "end": "H181"},
        {"grade": 3, "holds": ["H4", "H37", "H70", "H103", "H136", "H169"], "start": "H4", "end": "H169"},
        {"grade": 3, "holds": ["H5", "H27", "H61", "H93", "H127", "H159", "H192"], "start": "H5", "end": "H192"},
        {"grade": 4, "holds": ["H16", "H48", "H59", "H94", "H125", "H146", "H180"], "start": "H16", "end": "H180"},
        {"grade": 4, "holds": ["H3", "H38", "H72", "H104", "H138", "H171"], "start": "H3", "end": "H171"},
        {"grade": 4, "holds": ["H6", "H27", "H60", "H93", "H114", "H148", "H182"], "start": "H6", "end": "H182"},
        {"grade": 5, "holds": ["H14", "H49", "H80", "H115", "H146", "H179"], "start": "H14", "end": "H179"},
        {"grade": 5, "holds": ["H5", "H39", "H71", "H105", "H137", "H170"], "start": "H5", "end": "H170"},
        {"grade": 5, "holds": ["H16", "H37", "H70", "H91", "H126", "H157", "H190"], "start": "H16", "end": "H190"},
        {"grade": 5, "holds": ["H4", "H26", "H59", "H92", "H125", "H158", "H191"], "start": "H4", "end": "H191"}
    ]

    out_path = Path("data/raw/real_sample.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[make_real] Generating {len(raw_data)} routes...")

    with out_path.open("w", encoding="utf-8") as f:
        for i, raw in enumerate(raw_data):
            # 1. 转换 holds 为详细格式
            detailed_holds = []
            
            # 集合去重，并包含 start/end
            all_ids = set(raw["holds"])
            all_ids.add(raw["start"])
            all_ids.add(raw["end"])
            
            # 排序确保顺序一致性
            sorted_ids = sorted(list(all_ids), key=lambda x: int(x.replace("H", "")))

            for hid_str in sorted_ids:
                r, c = get_rc(hid_str)
                
                # 判定角色
                role = "M"
                if hid_str == raw["start"]: role = "S"
                elif hid_str == raw["end"]: role = "F"
                
                detailed_holds.append({"r": r, "c": c, "role": role})

            # 2. 构建最终记录
            rec = {
                "id": f"real_{i}",
                "grade": raw["grade"],
                "holds": detailed_holds, # 这里是 preprocess 喜欢的格式
                "board": "moonboard"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"[make_real] Done! Saved to {out_path}")

if __name__ == "__main__":
    make_real_data()