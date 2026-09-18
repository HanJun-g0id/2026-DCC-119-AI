import torch.nn as nn
from transformers import Wav2Vec2Model


class MeanPooling(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


class GenderClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name="facebook/wav2vec2-base",
        num_classes=2,
        freeze_feature_extractor=True,
        dropout=0.1,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        hidden_size = self.encoder.config.hidden_size
        self.pool = MeanPooling()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        feat_attention_mask = None
        if attention_mask is not None:
            feat_attention_mask = self.encoder._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )

        pooled = self.pool(hidden_states, feat_attention_mask)
        logits = self.classifier(pooled)
        return logits
