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
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.token2id = json.load(f)
            self.id2token = {int(v) if isinstance(v, str) and v.isdigit() else v: k for k, v in self.token2id.items()}
            self.vocab_size = len(self.token2id)
        def decode(self, ids):
            return [self.id2token.get(i, "<UNK>") for i in ids]

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f: 
        cfg = yaml.safe_load(f)

    run_dir = Path("outputs/phase2")
    subdirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and "action_model" in d.name], key=lambda d: d.stat().st_mtime, reverse=True)
    if not subdirs: 
        print("No model found!")
        return
        
    latest_ckpt_dir = subdirs[0]

    out_dir = run_dir / f"{latest_ckpt_dir.name.split('_')[0]}_{latest_ckpt_dir.name.split('_')[1]}_generate_action_{latest_ckpt_dir.name.split('_')[-1]}"
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ActionTokenizer(str(Path("data/processed_actions/action_tokenizer_vocab.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepRouteSet(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=cfg["model"]["embed_dim"], 
        hidden_dim=cfg["model"]["hidden_dim"], 
        num_layers=cfg["model"]["num_layers"], 
        dropout=cfg["model"].get("dropout", 0.0),
        pad_id=tokenizer.token2id.get("<PAD>", 0)
    )
    
    # === 核心修复：智能拆快递，完美读取模型权重 ===
    checkpoint = torch.load(latest_ckpt_dir / "action_model.pt", map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # ============================================
    
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
            bos_token = "<BOS>" if "<BOS>" in tokenizer.token2id else "<START>"
            seq_input = torch.tensor([[tokenizer.token2id.get(bos_token, 0)]], dtype=torch.long, device=device)

            out_ids = []
            lh_r, lh_c, rh_r, rh_c = -1, -1, -1, -1

            for _ in range(max_len):
                logits = model(c_input, seq_input)[0, -1, :] / temp
                
                mask = torch.zeros_like(logits)
                step = len(out_ids)
                
                last_hand = None
                if step > 0:
                    prev_tok = tokenizer.id2token[out_ids[-1]]
                    if "LH" in prev_tok: 
                        last_hand = "LH"
                    elif "RH" in prev_tok: 
                        last_hand = "RH"
                
                for tok, idx in tokenizer.token2id.items():
                    if tok in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<START>", "<END>"]: 
                        mask[idx] = -float('inf')
                        continue
                        
                    if step < 2:
                        if not tok.startswith("START_"): 
                            mask[idx] = -float('inf')
                        elif step == 1:
                            if last_hand == "LH" and "LH" in tok: mask[idx] = -float('inf')
                            if last_hand == "RH" and "RH" in tok: mask[idx] = -float('inf')
                        continue

                    if tok.startswith("START_"):
                        mask[idx] = -float('inf')
                        continue
                        
                    if "_H" in tok:
                        hand, hid_str = tok.split("_H")
                        r, c = board.from_id(int(hid_str))
                        is_lh = "LH" in hand
                        
                        if step >= 2 and last_hand is not None:
                            if last_hand == "LH" and is_lh: mask[idx] = -float('inf')
                            if last_hand == "RH" and not is_lh: mask[idx] = -float('inf')
                        
                        if is_lh and rh_r != -1 and math.hypot(r - rh_r, c - rh_c) > 6.0: 
                            mask[idx] = -float('inf')
                        if not is_lh and lh_r != -1 and math.hypot(r - lh_r, c - lh_c) > 6.0: 
                            mask[idx] = -float('inf')

                probs = torch.softmax(logits + mask, dim=-1)
                if torch.isnan(probs).any() or probs.sum() == 0: 
                    break
                    
                next_id = torch.multinomial(probs, num_samples=1).item()
                eos_token = "<EOS>" if "<EOS>" in tokenizer.token2id else "<END>"
                if next_id == tokenizer.token2id.get(eos_token, 1): 
                    break
                
                out_ids.append(next_id)
                seq_input = torch.cat([seq_input, torch.tensor([[next_id]], device=device)], dim=1)
                
                tok = tokenizer.id2token[next_id]
                if "_H" in tok:
                    r, c = board.from_id(int(tok.split("_H")[-1]))
                    if "LH" in tok: 
                        lh_r, lh_c = r, c
                    else: 
                        rh_r, rh_c = r, c

            action_tokens = tokenizer.decode(out_ids)
            holds = []
            for t in action_tokens:
                if "_H" in t:
                    hid = int(t.split("_H")[-1])
                    if not holds or holds[-1] != hid: 
                        holds.append(hid)
            
            if len(holds) >= 5:
                success_recs.append({"id": f"gen_{i:04d}", "grade": grade, "action_tokens": action_tokens, "seq_betamove": holds})

    print(f"[{__name__}] 强状态机交替生成完毕! 尝试生成 {num_gen} 条, 合法 {len(success_recs)} 条")
    with (art_dir / "action_generated_routes.jsonl").open("w", encoding="utf-8") as f:
        for r in success_recs: 
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()