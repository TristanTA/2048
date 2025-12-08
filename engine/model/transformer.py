import torch
import torch.nn as nn

from model.embedder import Embedder

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class Transformer2048(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=128,
                 num_layers=3, num_tile_types=16):
        super().__init__()

        self.grid_size = 16
        self.embedder = Embedder(self.grid_size, embed_dim, num_tile_types)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        # Q-value head: 4 actions (up, down, left, right)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.grid_size * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, tile_values):
        x = self.embedder(tile_values)   # [batch, 16, embed_dim]

        for layer in self.layers:
            x = layer(x)

        q_values = self.head(x)          # [batch, 4]
        return q_values