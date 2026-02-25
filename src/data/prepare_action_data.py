from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.tokenizer import build_action_tokenizer, save_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", default="data/processed_actions")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--cols", type=int, default=11)
    ap.add_argument("--max_grade", type=int, default=20)
    args = ap.parse_args()

    inp_dir = Path(args.inp_dir)
    
    # 1. 构建全新的 Action Tokenizer 词表
    print("[Prepare Action Data] 正在构建相对动作全集词表...")
    tok = build_action_tokenizer(args.rows, args.cols, args.max_grade)
    
    # 2. 保存词表
    vocab_path = inp_dir / "action_tokenizer_vocab.json"
    save_tokenizer(tok, str(vocab_path))
    print(f"[Prepare Action Data] 词表构建完成！词表大小: {len(tok.vocab)} -> 保存至 {vocab_path}")

    # 3. 将动作数据替换成统一的 "seq" 字段，供训练脚本直接读取
    splits = ["train", "val", "test"]
    for split in splits:
        action_file = inp_dir / f"{split}_actions.jsonl"
        final_file = inp_dir / f"{split}_final.jsonl"
        
        if not action_file.exists():
            continue
            
        lines = action_file.read_text(encoding="utf-8").splitlines()
        
        with final_file.open("w", encoding="utf-8") as f_out:
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                
                # 模型训练代码默认读取 'seq' 和 'grade'
                final_rec = {
                    "id": rec["id"],
                    "grade": rec["grade"],
                    "seq": rec["action_seq"]  # 把动作序列作为主要训练序列
                }
                f_out.write(json.dumps(final_rec, ensure_ascii=False) + "\n")
                
        print(f"[Prepare Action Data] 已打包 {split:5s} 训练集 -> {final_file}")

if __name__ == "__main__":
    main()