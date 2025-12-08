import torch
import torch.nn as nn
import numpy as np

class Embedder(nn.Module):
    def __init__(self, x_dim, y_dim, embed_dim, num_tile_types):
        super().__init__()
        self.grid_size = x_dim * y_dim
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

    def get_value_id_tensor(session):
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
            2048: 11
        }
        vals = [0] * (session.x_grid * session.y_grid)
        for sq in session.values:
            pos = session.get_position(sq)
            tile_val = sq.value
            vals[pos] = tile_to_id.get(tile_val, 0)

        return torch.tensor([vals], dtype=torch.long)
