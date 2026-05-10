import torch.nn as nn
import torch
from transformers import Blip2QFormerConfig, Blip2QFormerModel

class EmoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 77 * 768)
        )


    def forward(self, valence, arousal):
        va = torch.cat([valence, arousal], dim=-1)
        x = self.fc(va)
        x = x.view(-1, 77, 768)
       
        return x
    
class EmotionMapper(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()
        self.emo_encoder = EmoEncoder()

        qformer_config = Blip2QFormerConfig()
        qformer_config.encoder_hidden_size = 768
        qformer_config.attention_probs_dropout_prob = 0.0
        self.qformer = Blip2QFormerModel(qformer_config)
        self.layer_norm = nn.LayerNorm(text_dim, eps=1e-12)

    def forward(self, valence, arousal, image_embeds):

        emo_features = self.emo_encoder(valence, arousal)

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=image_embeds.dtype, device=image_embeds.device)

        
        query_outputs = self.qformer(
            query_embeds=emo_features,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]
        output = self.layer_norm(query_outputs)
        
        return output

