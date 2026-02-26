import argparse
import json
import math
from pathlib import Path

import torch
from src.env.board import Board
from src.models.action_model import ActionClimbLSTM
from src.data.action_tokenizer import ActionTokenizer

def decode_actions_to_holds(action_tokens: list[str], board: Board) -> list[int] | None:
    """
    带有严格生物力学约束的解码器。
    彻底消灭“一只手单吊”（双臂跨度过大）的无效路线！
    """
    holds = []
    lh_r, lh_c = -1, -1
    rh_r, rh_c = -1, -1
    max_reach = 6.0 # 绝对物理限制：双臂距离不能超过 6.0 格

    for tok in action_tokens:
        if tok.startswith("START_H"):
            hid_str = tok.split("H")[-1]
            if not hid_str.isdigit(): continue
            hid = int(hid_str)
            r, c = board.from_id(hid)
            
            if lh_r == -1 and rh_r == -1:
                lh_r, lh_c = r, c
                rh_r, rh_c = r, c
            elif r != lh_r or c != lh_c:
                if c > lh_c: rh_r, rh_c = r, c
                else: rh_r, rh_c, lh_r, lh_c = lh_r, lh_c, r, c
            holds.append(hid)
            
        elif "_R" in tok and "_C" in tok:
            try:
                parts = tok.split("_")
                hand = parts[0]  
                dr = int([p for p in parts if p.startswith('R') and (p[1] in '+-')][0][1:])
                dc = int([p for p in parts if p.startswith('C') and (p[1] in '+-')][0][1:])
            except Exception:
                continue
            
            if hand == "LH":
                if lh_r == -1: continue
                new_lh_r, new_lh_c = lh_r + dr, lh_c + dc
                
                # 1. 越界检查
                if not (0 <= new_lh_r < board.rows and 0 <= new_lh_c < board.cols): break
                # 2. 物理跨度检查：如果这只手伸过去，导致双手距离超过极限，立刻熔断！
                if rh_r != -1 and math.hypot(new_lh_r - rh_r, new_lh_c - rh_c) > max_reach: break
                
                lh_r, lh_c = new_lh_r, new_lh_c
                holds.append(board.to_id(lh_r, lh_c))
                
            elif hand == "RH":
                if rh_r == -1: continue
                new_rh_r, new_rh_c = rh_r + dr, rh_c + dc
                
                if not (0 <= new_rh_r < board.rows and 0 <= new_rh_c < board.cols): break
                if lh_r != -1 and math.hypot(lh_r - new_rh_r, lh_c - new_rh_c) > max_reach: break
                
                rh_r, rh_c = new_rh_r, new_rh_c
                holds.append(board.to_id(rh_r, rh_c))

    # 去重
    clean_holds = []
    for h in holds:
        if not clean_holds or clean_holds[-1] != h: clean_holds.append(h)

    return clean_holds

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 找到最新的 checkpoint
    run_dir = Path("outputs/phase2")
    if not run_dir.exists():
        print("No run dir found!")
        return

    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "action_model" in d.name]
    if not subdirs:
        print("No model found!")
        return
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_ckpt_dir = subdirs[0]
    ckpt_path = latest_ckpt_dir / "action_model.pt"

    print(f"[{__name__}] Loading checkpoint: {ckpt_path}")

    # 获取最新时间戳，用于创建新的 generate 目录
    ts = latest_ckpt_dir.name.split("_")[0] + "_" + latest_ckpt_dir.name.split("_")[1]
    hash_tag = latest_ckpt_dir.name.split("_")[-1]
    out_dir = run_dir / f"{ts}_generate_action_{hash_tag}"
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Tokenizer
    vocab_path = Path("data/processed_actions/action_tokenizer_vocab.json")
    tokenizer = ActionTokenizer(str(vocab_path))

    model_cfg = cfg["model"]
    model = ActionClimbLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    board = Board()
    num_gen = cfg["generate"]["num_samples"]
    max_len = cfg["generate"]["max_length"]
    temp = cfg["generate"]["temperature"]

    success_recs = []
    
    with torch.no_grad():
        for i in range(num_gen):
            grade = torch.randint(3, 7, (1,)).item()
            c_input = torch.tensor([[grade]], dtype=torch.long, device=device)
            start_tok = tokenizer.token2id.get("<START>", 0)
            seq_input = torch.tensor([[start_tok]], dtype=torch.long, device=device)

            out_ids = []
            for _ in range(max_len):
                logits = model(c_input, seq_input)
                next_logits = logits[0, -1, :] / temp
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                
                if next_id == tokenizer.token2id.get("<END>", 1):
                    break
                    
                out_ids.append(next_id)
                seq_input = torch.cat([seq_input, torch.tensor([[next_id]], device=device)], dim=1)

            action_tokens = tokenizer.decode(out_ids)
            holds = decode_actions_to_holds(action_tokens, board)
            
            if holds and len(holds) >= 4:
                rec = {
                    "id": f"gen_action_{i:04d}",
                    "grade": grade,
                    "action_tokens": action_tokens,
                    "seq_betamove": holds
                }
                success_recs.append(rec)

    print(f"[{__name__}] 生物力学生成完毕! 尝试生成 {num_gen} 条, 物理合法 {len(success_recs)} 条 (合法率 {len(success_recs)/num_gen*100:.1f}%)")
    
    out_file = art_dir / "action_generated_routes.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in success_recs:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()