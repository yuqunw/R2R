import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1024,
        geometry_dim: int = 2048,
        attn_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert attn_dim == vision_dim, (
            "attn_dim must equal vision_dim so the residual connection works "
            "without an extra projection. If you need a different attn_dim, "
            "add an output projection back to vision_dim."
        )

        num_geometry_layers = len(os.environ.get('VGGT_MULTI_LAYER_INDICES', '4,11,17,23').split(','))
        self.patch_mapping = nn.Sequential(
                nn.LayerNorm(geometry_dim * num_geometry_layers),
                nn.Linear(geometry_dim * num_geometry_layers, geometry_dim),
            )
        self.norm_q = nn.LayerNorm(vision_dim)
        self.norm_kv = nn.LayerNorm(geometry_dim)

        self.q_proj = nn.Linear(vision_dim, attn_dim)
        self.k_proj = nn.Linear(geometry_dim, attn_dim)
        self.v_proj = nn.Linear(geometry_dim, attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(vision_dim)
        self.mlp = GatedMLP(
            dim=vision_dim,
            hidden_dim=int(vision_dim * mlp_ratio),
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.mlp.down_proj.weight)

    def forward(
        self,
        vision_features: torch.Tensor,
        geometry_camera_embeds: torch.Tensor,
        geometry_patch_embeds: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ):
        mapped_patch_embeds = self.patch_mapping(geometry_patch_embeds)
        geometry_features = torch.cat([geometry_camera_embeds, mapped_patch_embeds], dim=1)
        q = self.q_proj(self.norm_q(vision_features))
        kv_normed = self.norm_kv(geometry_features)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)

        attn_out, attn_weights = self.cross_attn(
            query=q, key=k, value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
        )
        x = vision_features + self.attn_dropout(attn_out)

        x = x + self.mlp(self.norm_mlp(x))

        return x, attn_weights
