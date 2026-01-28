from __future__ import annotations
import torch
import torch.nn as nn

class DeepRouteSet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, pad_id: int, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        # 升级点 1: 使用 LSTM 替代 GRU，捕捉更长距离的攀爬逻辑
        # 升级点 2: 增加 Dropout，防止模型对 1000 条数据过拟合
        self.rnn = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout) # 输出层前的额外 Dropout

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: [B,L]
        emb = self.embed(x_in)
        
        # LSTM 返回 (output, (h_n, c_n))，我们需要 output
        out, _ = self.rnn(emb) 
        
        # 经过 Dropout 和全连接层
        out = self.dropout(out)
        logits = self.lm_head(out)  # [B,L,V]
        return logits

    @torch.no_grad()
    def generate(self, bos: int, eos: int, prefix: list[int], max_len: int, temperature: float = 1.0, top_k: int = 0) -> list[int]:
        device = next(self.parameters()).device
        seq = prefix[:]
        
        # 自回归生成循环
        while len(seq) < max_len:
            x = torch.tensor([seq], dtype=torch.long, device=device)
            
            # 获取最后一个时间步的预测
            logits = self.forward(x)[:, -1, :] 
            
            # 温度采样 (Temperature Scaling)
            logits = logits / max(temperature, 1e-6)
            
            # Top-k 采样 (截断低概率尾部)
            if top_k and top_k > 0:
                v, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(v, dim=-1)
                next_idx_in_topk = torch.multinomial(probs[0], 1).item()
                next_id = idx[0, next_idx_in_topk].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs[0], 1).item()
            
            seq.append(int(next_id))
            if next_id == eos:
                break
        return seq