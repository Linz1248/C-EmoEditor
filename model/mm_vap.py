import torch
import torch.nn as nn
from typing import List, Optional

# def freeze_module(module: nn.Module):
#     for p in module.parameters():
#         p.requires_grad = False

class CrossAttentionBlock(nn.Module):
    def __init__(self, img_dim: int, text_dim: int, bridge_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.bridge_dim = bridge_dim
        self.q_proj = nn.Linear(img_dim, bridge_dim) if img_dim != bridge_dim else nn.Identity()
        self.k_proj = nn.Linear(text_dim, bridge_dim) if text_dim != bridge_dim else nn.Identity()
        self.v_proj = nn.Linear(text_dim, bridge_dim) if text_dim != bridge_dim else nn.Identity()

        self.mha = nn.MultiheadAttention(embed_dim=bridge_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.norm1 = nn.LayerNorm(bridge_dim)
        self.ffn = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim * 4),
            nn.GELU(),
            nn.Linear(bridge_dim * 4, bridge_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(bridge_dim)

    def forward(self, img_feats: torch.Tensor, text_feats: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        Q = self.q_proj(img_feats)
        K = self.k_proj(text_feats)
        V = self.v_proj(text_feats)

        attn_out, _ = self.mha(Q, K, V, key_padding_mask=text_mask)
        x = self.norm1(Q + attn_out)
        x2 = self.ffn(x)
        out = self.norm2(x + x2)
        return out


class LayerAttentionAggregator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, layer_stack: torch.Tensor):
        scores = self.attention_weights(layer_stack)
        alpha = torch.softmax(scores, dim=2)
        fused = (alpha * layer_stack).sum(dim=2)
        return fused


class AttentiveRegressionHead(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.3, attention_layers: int = 4):
        super().__init__()

        self.valence_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )

        self.arousal_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        
        self.attention_proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        
        instance_valence_preds = self.valence_predictor(x)
        instance_arousal_preds = self.arousal_predictor(x)
        instance_preds = torch.cat([instance_valence_preds, instance_arousal_preds], dim=-1)
        
        context_features = self.attention_encoder(x) 
        
        att_logits = self.attention_proj(context_features)
        
        att_weights = torch.softmax(att_logits, dim=1)
        
        att_weights = torch.nan_to_num(att_weights, nan=0.0)

        final_pred = torch.sum(instance_preds * att_weights, dim=1)
        
        return final_pred


class CrossAttentionBridge(nn.Module):
    def __init__(self, img_dims: List[int], text_dims: List[int], bridge_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        L = min(len(img_dims), len(text_dims))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(img_dim=img_dims[i], text_dim=text_dims[i], bridge_dim=bridge_dim, num_heads=num_heads, dropout=dropout)
            for i in range(L)
        ])
        
        self.layer_aggregator = LayerAttentionAggregator(bridge_dim)
        self.final_proj = nn.Linear(bridge_dim, bridge_dim)
        self.norm = nn.LayerNorm(bridge_dim)

    def forward(self, img_hidden_states: List[torch.Tensor], text_hidden_states: List[torch.Tensor], text_mask: Optional[torch.Tensor] = None):
        L = min(len(img_hidden_states), len(text_hidden_states))
        outputs = []
        for i in range(L):
            img_h = img_hidden_states[i]
            text_h = text_hidden_states[i]
            out = self.blocks[i](img_h, text_h, text_mask=text_mask)
            outputs.append(out)

        stacked_outputs = torch.stack(outputs, dim=2)
        fused_tokens = self.layer_aggregator(stacked_outputs)
        
        fused_tokens = self.final_proj(fused_tokens)
        fused_tokens = self.norm(fused_tokens)
        return fused_tokens


class VAPredictor(nn.Module):
    def __init__(self, vision_config, text_config, bridge_dim: int = 512, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()

        num_vis_layers = vision_config.num_hidden_layers + 1
        num_txt_layers = text_config.num_hidden_layers + 1
        
        img_dims = [vision_config.hidden_size] * num_vis_layers
        text_dims = [text_config.hidden_size] * num_txt_layers

        self.bridge = CrossAttentionBridge(img_dims=img_dims, text_dims=text_dims, bridge_dim=bridge_dim, num_heads=num_heads, dropout=dropout)

        self.regressor = AttentiveRegressionHead(
            input_dim=bridge_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            attention_layers=4
        )

    def forward(self, image_hidden_states: list, text_hidden_states: list, attention_mask: Optional[torch.Tensor] = None):

        text_key_padding_mask = None
        if attention_mask is not None:
            text_key_padding_mask = (attention_mask == 0)

        fused_sequence = self.bridge(
            img_hidden_states=image_hidden_states, 
            text_hidden_states=text_hidden_states, 
            text_mask=text_key_padding_mask
        )
        
        preds = self.regressor(fused_sequence)

        return preds