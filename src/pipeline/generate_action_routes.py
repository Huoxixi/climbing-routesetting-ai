import argparse
import json
import math
from pathlib import Path
import torch
from src.env.board import Board
from src.models.deeprouteset import DeepRouteSet

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    if A == C or A == D or B == C or B == D: return False
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

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

            if grade <= 4: dyn_max_reach, dyn_finish_r = 4.0, 14
            else: dyn_max_reach, dyn_finish_r = 6.0, 16

            out_ids = []
            segments = [] # 记录推理过程中的所有线段
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
                            mask[idx] = -float('inf')
                            continue
                            
                        if step < 2:
                            if not tok.startswith("START_"): mask[idx] = -float('inf')
                            elif step == 1:
                                if "LH" in prev_tok and "LH" in tok: mask[idx] = -float('inf')
                                if "RH" in prev_tok and "RH" in tok: mask[idx] = -float('inf')
                            continue

                        if tok.startswith("START_"):
                            mask[idx] = -float('inf')
                            continue
                            
                        if "_H" in tok:
                            hand, hid_str = tok.split("_H")
                            r, c = board.from_id(int(hid_str))
                            is_lh = "LH" in hand
                            
                            cur_r = lh_r if is_lh else rh_r
                            cur_c = lh_c if is_lh else rh_c
                            static_r, static_c = (rh_r, rh_c) if is_lh else (lh_r, lh_c)
                            
                            if step >= 2 and last_hand is not None:
                                if last_hand == "LH" and is_lh: mask[idx] = -float('inf')
                                if last_hand == "RH" and not is_lh: mask[idx] = -float('inf')
                            
                            if static_r != -1:
                                if r <= cur_r: mask[idx] = -float('inf') # 向上
                                if is_lh and c > static_c + 1: mask[idx] = -float('inf') # 防交叉
                                if not is_lh and c < static_c - 1: mask[idx] = -float('inf')
                                if math.hypot(r - static_r, c - static_c) > dyn_max_reach: mask[idx] = -float('inf')

                                # 【几何雷达：如果这条新线段与历史任意线段交叉，无情封杀！】
                                if cur_r != -1:
                                    is_crossing = False
                                    for seg in segments:
                                        if intersect((cur_r, cur_c), (r, c), seg[0], seg[1]):
                                            is_crossing = True
                                            break
                                    if is_crossing: mask[idx] = -float('inf')

                probs = torch.softmax(logits + mask, dim=-1)
                if torch.isnan(probs).any() or probs.sum() == 0: break
                    
                next_id = torch.multinomial(probs, num_samples=1).item()
                if next_id == eos_token_id: break
                
                out_ids.append(next_id)
                seq_input = torch.cat([seq_input, torch.tensor([[next_id]], device=device)], dim=1)
                
                tok = tokenizer.id2token[next_id]
                if "_H" in tok:
                    r, c = board.from_id(int(tok.split("_H")[-1]))
                    is_lh = "LH" in tok
                    prev_r = lh_r if is_lh else rh_r
                    prev_c = lh_c if is_lh else rh_c
                    if prev_r != -1: segments.append(((prev_r, prev_c), (r, c))) # 记录成功走过的线段
                    
                    if is_lh: lh_r, lh_c = r, c
                    else: rh_r, rh_c = r, c

            action_tokens = tokenizer.decode(out_ids)
            holds = []
            for t in action_tokens:
                if "_H" in t:
                    hid = int(t.split("_H")[-1])
                    if not holds or holds[-1] != hid: holds.append(hid)
            
            if len(holds) >= 5:
                success_recs.append({"id": f"gen_{i:04d}", "grade": grade, "action_tokens": action_tokens, "seq_betamove": holds})

    print(f"[{__name__}] 几何防交叉生成完毕! 尝试生成 {num_gen} 条, 合法 {len(success_recs)} 条")
    with (art_dir / "action_generated_routes.jsonl").open("w", encoding="utf-8") as f:
        for r in success_recs: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()