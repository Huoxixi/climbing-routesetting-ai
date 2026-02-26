import argparse
import json
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed_actions")
    args = ap.parse_args()
    work_dir = Path(args.dir)
    
    token_counter = Counter()
    for split in ["train", "val", "test"]:
        file_path = work_dir / f"{split}_actions.jsonl"
        if not file_path.exists(): continue
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip(): 
                rec = json.loads(line)
                token_counter.update(rec.get("action_seq", []))
                
    # 【终极兜底】：强行注入旧框架死活要找的 <BOS> 和 <EOS>
    vocab = ["<PAD>", "<BOS>", "<EOS>", "<START>", "<END>", "<UNK>"] + [tok for tok, _ in token_counter.most_common()]
    token2id = {t: i for i, t in enumerate(vocab)}
    
    vocab_path = work_dir / "action_tokenizer_vocab.json"
    # 强制粉碎旧的毒词表！
    if vocab_path.exists():
        vocab_path.unlink()
        
    vocab_path.write_text(json.dumps(token2id, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Prepare Data] 旧词表已粉碎，新词表注入 <BOS> 成功！大小: {len(vocab)}")

    for split in ["train", "val", "test"]:
        inp_file = work_dir / f"{split}_actions.jsonl"
        if not inp_file.exists(): continue
        processed = 0
        with (work_dir / f"{split}_final.jsonl").open("w", encoding="utf-8") as f:
            for line in inp_file.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                rec = json.loads(line)
                rec["input_ids"] = [token2id.get(tok, token2id["<UNK>"]) for tok in rec.get("action_seq", [])]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1
        print(f"[Prepare Data] 打包 {split} -> {processed} 条")

if __name__ == "__main__":
    main()