from __future__ import annotations
import torch
import torch.nn as nn

class DeepRouteSet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, pad_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: [B,L] input tokens
        emb = self.embed(x_in)
        out, _ = self.rnn(emb)
        logits = self.lm_head(out)  # [B,L,V]
        return logits

    @torch.no_grad()
    def generate(self, bos: int, eos: int, prefix: list[int], max_len: int, temperature: float = 1.0, top_k: int = 0) -> list[int]:
        device = next(self.parameters()).device
        seq = prefix[:]  # already includes BOS and maybe grade token
        while len(seq) < max_len:
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self.forward(x)[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(v, dim=-1)
                next_id = idx[0, torch.multinomial(probs[0], 1).item()].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs[0], 1).item()
            seq.append(int(next_id))
            if next_id == eos:
                break
        return seq
