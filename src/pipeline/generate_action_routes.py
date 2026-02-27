import argparse
import json
import math
from pathlib import Path
import torch
from src.env.board import Board
from src.models.deeprouteset import DeepRouteSet

try:
    from src.data.tokenizer import ActionTokenizer
except ImportError:
    class ActionTokenizer:
        def __init__(self, vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f: self.token2id = json.load(f)
            self.id2token = {int(v) if isinstance(v, str) and v.isdigit() else v: k for k, v in self.token2id.items()}
            self.vocab_size = len(self.token2id)
        def decode(self, ids): return [self.id2token.get(i, "<UNK>") for i in ids]

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f)

    run_dir = Path("outputs/phase2")
    valid_subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "action_model" in d.name and (d / "action_model.pt").exists()]
    if not valid_subdirs: return print("No model found!")
        
    valid_subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_ckpt_dir = valid_subdirs[0]

    out_dir = run_dir / f"{latest_ckpt_dir.name.split('_')[0]}_{latest_ckpt_dir.name.split('_')[1]}_generate_action_{latest_ckpt_dir.name.split('_')[-1]}"
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ActionTokenizer(str(Path("data/processed_actions/action_tokenizer_vocab.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepRouteSet(
        vocab_size=tokenizer.vocab_size, embed_dim=cfg["model"]["embed_dim"], 
        hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"], 
        dropout=cfg["model"].get("dropout", 0.0), pad_id=tokenizer.token2id.get("<PAD>", 0)
    )
    
    checkpoint = torch.load(latest_ckpt_dir / "action_model.pt", map_location=device)
    if "state_dict" in checkpoint: model.load_state_dict(checkpoint["state_dict"])
    else: model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    board = Board()
    num_gen = cfg["generation"]["samples_per_grade"] * 4  
    max_len = cfg["data"]["max_seq_len"]                  
    temp = cfg["generation"]["temperature"]               
    
    success_recs = []
    eos_token_id = tokenizer.token2id.get("<EOS>", tokenizer.token2id.get("<END>", 1))

    with torch.no_grad():
        for i in range(num_gen):
            grade = torch.randint(3, 7, (1,)).item()
            bos_token = "<BOS>" if "<BOS>" in tokenizer.token2id else "<START>"
            seq_input = torch.tensor([[tokenizer.token2id.get(bos_token, 0)]], dtype=torch.long, device=device)

            if grade <= 4: dyn_max_reach, dyn_finish_r = 4.0, 15
            else: dyn_max_reach, dyn_finish_r = 6.0, 17

            out_ids = []
            lh_r, lh_c, rh_r, rh_c = -1, -1, -1, -1

            for _ in range(max_len):
                logits = model(seq_input)[0, -1, :] / temp
                mask = torch.zeros_like(logits)
                step = len(out_ids)
                
                if max(lh_r, rh_r) >= dyn_finish_r:
                    mask[:] = -float('inf')
                    mask[eos_token_id] = 0.0  
                else:
                    last_hand = None
                    if step > 0:
                        prev_tok = tokenizer.id2token[out_ids[-1]]
                        if "LH" in prev_tok: last_hand = "LH"
                        elif "RH" in prev_tok: last_hand = "RH"
                    
                    for tok, idx in tokenizer.token2id.items():
                        if tok in ["<PAD>", "<UNK>", "<BOS>", "<START>"]: 
                            mask[idx] = -float('inf'); continue
                            
                        # =========================================================
                        # 🚀 起步 1 和 2 点的动态推理强掩码
                        # =========================================================
                        if step < 2:
                            if not tok.startswith("START_"): mask[idx] = -float('inf'); continue
                            
                            if "_H" in tok:
                                _, hid_str = tok.split("_H")
                                r, c = board.from_id(int(hid_str))
                                
                                # 强制 1、2 号点必须在 2 或 3 层
                                if r not in [2, 3]: mask[idx] = -float('inf')
                                
                                # 对于第 2 号点，验证与 1 号点的几何关系
                                if step == 1:
                                    if last_hand == "LH" and "LH" in tok: mask[idx] = -float('inf')
                                    if last_hand == "RH" and "RH" in tok: mask[idx] = -float('inf')
                                    
                                    prev_r, prev_c = board.from_id(int(prev_tok.split("_H")[-1]))
                                    
                                    # 纵坐标差距 0-1
                                    if abs(r - prev_r) > 1: mask[idx] = -float('inf')
                                    
                                    # 横坐标差距 < 3 且不共点 (1 或 2格)
                                    if abs(c - prev_c) >= 3 or abs(c - prev_c) == 0: mask[idx] = -float('inf')
                                    
                                    # 左手在左，右手在右
                                    if "LH" in tok and c >= prev_c: mask[idx] = -float('inf')
                                    if "RH" in tok and c <= prev_c: mask[idx] = -float('inf')
                            continue

                        if tok.startswith("START_"): mask[idx] = -float('inf'); continue

                        if "_H" in tok:
                            hand, hid_str = tok.split("_H")
                            r, c = board.from_id(int(hid_str))
                            is_lh = "LH" in hand
                            
                            cur_r = lh_r if is_lh else rh_r
                            static_r, static_c = (rh_r, rh_c) if is_lh else (lh_r, lh_c)
                            
                            if step >= 2 and last_hand is not None:
                                if last_hand == "LH" and is_lh: mask[idx] = -float('inf')
                                if last_hand == "RH" and not is_lh: mask[idx] = -float('inf')
                            
                            if static_r != -1:
                                if r <= cur_r: mask[idx] = -float('inf') 
                                if is_lh and c >= static_c: mask[idx] = -float('inf') 
                                if not is_lh and c <= static_c: mask[idx] = -float('inf')
                                if math.hypot(r - static_r, c - static_c) > dyn_max_reach: mask[idx] = -float('inf')

                probs = torch.softmax(logits + mask, dim=-1)
                if torch.isnan(probs).any() or probs.sum() == 0: break
                    
                next_id = torch.multinomial(probs, num_samples=1).item()
                if next_id == eos_token_id: break
                
                out_ids.append(next_id)
                seq_input = torch.cat([seq_input, torch.tensor([[next_id]], device=device)], dim=1)
                
                tok = tokenizer.id2token[next_id]
                if "_H" in tok:
                    r, c = board.from_id(int(tok.split("_H")[-1]))
                    if "LH" in tok: lh_r, lh_c = r, c
                    else: rh_r, rh_c = r, c

            action_tokens = tokenizer.decode(out_ids)
            holds = []
            for t in action_tokens:
                if "_H" in t:
                    hid = int(t.split("_H")[-1])
                    if not holds or holds[-1] != hid: holds.append(hid)
            
            if len(holds) >= 5:
                first_r, first_c = board.from_id(holds[0])
                sec_r, sec_c = board.from_id(holds[1])
                
                # 🚀 重心计算：B点自动落在 1号 和 2号 正中央 (或者其中某一个正下方)
                base_c = (first_c + sec_c) // 2
                base_holds = [board.to_id(0, base_c)]
                
                finish_holds = [holds[-1]]
                start_move_holds = [holds[0], holds[1]]
                
                success_recs.append({
                    "id": f"gen_{i:04d}", "grade": grade, 
                    "base_holds": base_holds, "finish_holds": finish_holds,
                    "start_move_holds": start_move_holds,
                    "action_seq": action_tokens, "seq_betamove": holds
                })

    with (art_dir / "action_generated_routes.jsonl").open("w", encoding="utf-8") as f:
        for r in success_recs: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()