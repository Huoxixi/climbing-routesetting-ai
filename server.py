import os
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import yaml
import uvicorn

# 导入你项目现有的核心模块
from src.env.board import Board
from src.models.deeprouteset import DeepRouteSet
from src.data.tokenizer import load_tokenizer  

app = FastAPI(title="AI Climbing Route Generator API")

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 恢复接收难度的请求体
class GradeRequest(BaseModel):
    grade: int  

# 全局变量，用于在应用启动时加载模型缓存
AI_ENGINE = {}

@app.on_event("startup")
async def load_ai_model():
    print("⏳ 正在加载底层物理环境与 AI 模型，请稍候...")
    try:
        AI_ENGINE['board'] = Board()
        tokenizer_path = Path("data/processed_actions/action_tokenizer_vocab.json")
        AI_ENGINE['tokenizer'] = load_tokenizer(str(tokenizer_path))  
        
        with open("configs/phase2.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        AI_ENGINE['cfg'] = cfg
        
        run_dir = Path("outputs/phase2")
        valid_subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "action_model" in d.name and (d / "action_model.pt").exists()]
        if not valid_subdirs:
            raise FileNotFoundError("在 outputs/phase2/ 目录下没有找到有效的 action_model.pt 权重！")
        
        valid_subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        latest_ckpt_dir = valid_subdirs[0]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepRouteSet(
            vocab_size=len(AI_ENGINE['tokenizer'].vocab), 
            embed_dim=cfg["model"]["embed_dim"], 
            hidden_dim=cfg["model"]["hidden_dim"], 
            num_layers=cfg["model"]["num_layers"], 
            pad_id=AI_ENGINE['tokenizer'].vocab.get("<PAD>", 0)
        )
        
        ckpt_path = latest_ckpt_dir / "action_model.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        AI_ENGINE['model'] = model
        AI_ENGINE['device'] = device
        print(f"✅ AI 模型加载成功！使用设备: {device}, 权重路径: {ckpt_path.name}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

def coord_id_to_string(hid):
    r, c = AI_ENGINE['board'].from_id(hid)
    col_letter = chr(65 + c) 
    return f"{col_letter}{r}"

@app.post("/generate_route")
async def generate_route_api(req: GradeRequest):
    if 'model' not in AI_ENGINE:
        raise HTTPException(status_code=500, detail="AI 模型未成功加载，请查看后端终端报错。")
    
    print(f"📥 收到生成请求：难度 V{req.grade}")
    
    model = AI_ENGINE['model']
    tokenizer = AI_ENGINE['tokenizer']
    board = AI_ENGINE['board']
    device = AI_ENGINE['device']
    cfg = AI_ENGINE['cfg']
    
    max_len = cfg["data"]["max_seq_len"]
    temp = cfg["generation"]["temperature"]
    eos_token_id = tokenizer.vocab.get("<EOS>", tokenizer.vocab.get("<END>", 1))
    
    # 🌟 保留 V0-V2 的新手友好物理限制
    grade = req.grade
    if grade <= 2:
        dyn_max_reach, dyn_finish_r = 3.5, 14
    elif grade <= 4:
        dyn_max_reach, dyn_finish_r = 4.0, 15
    else:
        dyn_max_reach, dyn_finish_r = 6.0, 17
        
    bos_token = "<BOS>" if "<BOS>" in tokenizer.vocab else "<START>"
    seq_input = torch.tensor([[tokenizer.vocab.get(bos_token, 0)]], dtype=torch.long, device=device)
    
    out_ids = []
    lh_r, lh_c, rh_r, rh_c = -1, -1, -1, -1
    
    with torch.no_grad():
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
                    prev_tok = tokenizer.ivocab[out_ids[-1]]
                    if "LH" in prev_tok: last_hand = "LH"
                    elif "RH" in prev_tok: last_hand = "RH"
                
                for tok, idx in tokenizer.vocab.items():
                    if tok in ["<PAD>", "<UNK>", "<BOS>", "<START>"]: 
                        mask[idx] = -float('inf'); continue
                        
                    if step < 2:
                        if not tok.startswith("START_"): mask[idx] = -float('inf'); continue
                        if "_H" in tok:
                            _, hid_str = tok.split("_H")
                            r, c = board.from_id(int(hid_str))
                            if r not in [2, 3]: mask[idx] = -float('inf')
                            if step == 1:
                                if last_hand == "LH" and "LH" in tok: mask[idx] = -float('inf')
                                if last_hand == "RH" and "RH" in tok: mask[idx] = -float('inf')
                                prev_r, prev_c = board.from_id(int(prev_tok.split("_H")[-1]))
                                if abs(r - prev_r) > 1: mask[idx] = -float('inf')
                                if abs(c - prev_c) >= 3 or abs(c - prev_c) == 0: mask[idx] = -float('inf')
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
            
            tok = tokenizer.ivocab[next_id]
            if "_H" in tok:
                r, c = board.from_id(int(tok.split("_H")[-1]))
                if "LH" in tok: lh_r, lh_c = r, c
                else: rh_r, rh_c = r, c

    action_tokens = tokenizer.decode(out_ids)
    frontend_route = []
    holds_ids = []
    
    for t in action_tokens:
        if "_H" in t:
            hid = int(t.split("_H")[-1])
            holds_ids.append(hid)
            hand_type = "LH" if "LH" in t else "RH"
            is_start = t.startswith("START_")
            frontend_route.append({
                "coord": coord_id_to_string(hid),
                "hand": hand_type,
                "is_start": is_start
            })

    base_holds_str = []
    if len(holds_ids) >= 2:
        first_r, first_c = board.from_id(holds_ids[0])
        sec_r, sec_c = board.from_id(holds_ids[1])
        base_c = (first_c + sec_c) // 2
        base_holds_str = [coord_id_to_string(board.to_id(0, base_c))] 

    finish_holds_str = [coord_id_to_string(holds_ids[-1])] if holds_ids else []

    print(f"✅ 路线生成完毕！总步数: {len(frontend_route)}")
    return {
        "status": "success",
        "route": frontend_route,
        "base_holds": base_holds_str,
        "finish_holds": finish_holds_str
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)