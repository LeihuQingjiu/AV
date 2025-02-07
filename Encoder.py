import torch
import torch.nn as nn


class SinPositionEncoding(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        sequence_length = x.size(0)  # 获取输入特征的序列长度
        pe = torch.zeros(sequence_length, self.d_model, dtype=torch.float).to(x.device)  # size(sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  # size(sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos

        return x + pe



# 视频编码器
class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(AudioEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = SinPositionEncoding(hidden_dim)
        self.channel_embedding = nn.Embedding(2, hidden_dim)  # 2代表左右两个音频通道
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

    def forward(self, unmasked_audio):
        audio_features = self.linear(unmasked_audio)
        batch_size, seq_length, _ = audio_features.size()

        # 添加正弦位置嵌入
        audio_features = self.positional_encoding(audio_features)

        # 添加可学习的通道嵌入
        channel_indices = torch.zeros(batch_size, seq_length, dtype=torch.long).to(audio_features.device)
        channel_embeddings = self.channel_embedding(channel_indices)
        audio_features = audio_features + channel_embeddings

        audio_features = audio_features.permute(1, 0, 2)  # Transformer要求的输入形状 (seq_len, batch_size, d_model)
        audio_features = self.transformer_encoder(audio_features)
        audio_features = audio_features.permute(1, 0, 2)
        return audio_features


class VisualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(VisualEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = SinPositionEncoding(hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

    def forward(self, visual_tokens):
        visual_features = self.linear(visual_tokens)

        # 添加正弦位置嵌入
        visual_features = self.positional_encoding(visual_features)

        visual_features = visual_features.permute(1, 0, 2)  # Transformer要求的输入形状 (seq_len, batch_size, d_model)
        visual_features = self.transformer_encoder(visual_features)
        visual_features = visual_features.permute(1, 0, 2)
        return visual_features


class SharedAVEncoder(nn.Module):
    def __init__(self, visual_feature_dim, audio_feature_dim, hidden_dim, num_layers):
        super(SharedAVEncoder, self).__init__()
        self.visual_positional_encoding = SinPositionEncoding(visual_feature_dim)
        self.audio_positional_encoding = SinPositionEncoding(audio_feature_dim)
        self.channel_embedding = nn.Embedding(2, audio_feature_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

    def forward(self, visual_features, audio_features):
        # 拼接视觉和音频特征
        combined_features = torch.cat((visual_features, audio_features), dim=1)

        # 分别添加正弦位置嵌入
        batch_size, seq_length, _ = combined_features.size()
        visual_seq_length = visual_features.size(1)
        audio_seq_length = audio_features.size(1)

        visual_part = combined_features[:, :visual_seq_length, :]
        audio_part = combined_features[:, visual_seq_length:, :]

        visual_part = self.visual_positional_encoding(visual_part)
        audio_part = self.audio_positional_encoding(audio_part)

        combined_features = torch.cat((visual_part, audio_part), dim=1)

        # 为音频特征添加通道嵌入
        channel_indices = torch.zeros(batch_size, audio_seq_length, dtype=torch.long).to(combined_features.device)
        channel_embeddings = self.channel_embedding(channel_indices)
        combined_features[:, visual_seq_length:, :] += channel_embeddings

        # 为所有特征添加可学习的模态嵌入
        modality_indices_visual = torch.zeros(batch_size, visual_seq_length, dtype=torch.long).to(combined_features.device)
        modality_indices_audio = torch.ones(batch_size, audio_seq_length, dtype=torch.long).to(combined_features.device)
        modality_indices = torch.cat((modality_indices_visual, modality_indices_audio), dim=1)

        modality_embeddings = self.modality_embedding(modality_indices)
        combined_features += modality_embeddings

        combined_features = combined_features.permute(1, 0, 2)
        av_features = self.transformer_encoder(combined_features)
        av_features = av_features.permute(1, 0, 2)

        return av_features



