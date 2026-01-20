from __future__ import annotations
import torch
import torch.nn as nn

class GradeNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, num_classes: int, pad_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # x: [B,L], attn: [B,L]
        emb = self.embed(x)
        out, _ = self.rnn(emb)  # [B,L,2H]
        # masked mean pooling
        mask = attn.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        logits = self.head(pooled)
        return logits
