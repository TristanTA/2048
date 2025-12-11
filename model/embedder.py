import torch
import torch.nn as nn
import numpy as np

class Embedder(nn.Module):
    def __init__(self, embed_dim, num_tile_types):
        super().__init__()
        self.grid_size = 16
        self.embed_dim = embed_dim
        self.value_embed = nn.Embedding(num_tile_types, embed_dim)
        self.pos_embed = nn.Embedding(self.grid_size, embed_dim)

    def forward(self, tile_values):
        batch, size = tile_values.shape
        pos_ids = torch.arange(size, device=tile_values.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch, size)
        val_emb = self.value_embed(tile_values)
        pos_emb = self.pos_embed(pos_ids)

        return val_emb + pos_emb

    def get_value_id_tensor(self, session):
        tile_to_id = {
            0: 0,
            2: 1,
            4: 2,
            8: 3,
            16: 4,
            32: 5,
            64: 6,
            128: 7,
            256: 8,
            512: 9,
            1024: 10,
            2048: 11,
            4096: 12,
            8192: 13,
            16384: 14,
            32768: 15,
            65536: 16,
            131072: 17
        }
        flat = session.board.flatten()
        vals = [tile_to_id.get(int(v), 0) for v in flat]

        return torch.tensor([vals], dtype=torch.long)