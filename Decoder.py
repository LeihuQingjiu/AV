import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x




class SharedAVDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SharedAVDecoder, self).__init__()
        self.linear_projection = nn.Linear(input_dim, hidden_dim)
        self.masked_token_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hidden_dim)
        self.visual_positional_encoding = PositionalEncoding(hidden_dim)
        self.audio_positional_encoding = PositionalEncoding(hidden_dim)
        self.channel_embedding = nn.Embedding(2, hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

    def forward(self, f_AV):
        # 创建低维投影
        g_AV = self.linear_projection(f_AV)

        # 假设S个掩码音频令牌，这里简单示例为1个，实际需根据情况调整
        S = 1
        masked_token_embeddings = self.masked_token_embedding(torch.zeros(S, dtype=torch.long).to(g_AV.device))
        g_AV = torch.cat((g_AV, masked_token_embeddings), dim=0)

        # 添加位置嵌入、通道嵌入和模态嵌入
        batch_size, seq_length, _ = g_AV.size()
        visual_seq_length = int(seq_length / 2)
        audio_seq_length = seq_length - visual_seq_length

        visual_part = g_AV[:visual_seq_length, :]
        audio_part = g_AV[visual_seq_length:, :]

        visual_part = self.visual_positional_encoding(visual_part)
        audio_part = self.audio_positional_encoding(audio_part)

        channel_indices = torch.zeros(audio_seq_length, dtype=torch.long).to(audio_part.device)
        channel_embeddings = self.channel_embedding(channel_indices)
        audio_part += channel_embeddings

        modality_indices_visual = torch.zeros(visual_seq_length, dtype=torch.long).to(visual_part.device)
        modality_indices_audio = torch.ones(audio_seq_length, dtype=torch.long).to(audio_part.device)
        modality_indices = torch.cat((modality_indices_visual, modality_indices_audio), dim=0)
        modality_embeddings = self.modality_embedding(modality_indices)
        g_AV = torch.cat((visual_part, audio_part), dim=0)
        g_AV += modality_embeddings

        # 输入浅层Transformer解码器
        g_AV = g_AV.unsqueeze(1)
        h_AV = self.transformer_decoder(g_AV)
        h_AV = h_AV.squeeze(1)

        # 提取音频特征
        h_A = h_AV[visual_seq_length:, :]
        return h_A


class AudioDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(AudioDecoder, self).__init__()
        self.audio_positional_encoding = PositionalEncoding(hidden_dim)
        self.channel_embedding = nn.Embedding(2, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

    def forward(self, g_A):
        # 重新添加位置嵌入和通道嵌入
        batch_size, seq_length, _ = g_A.size()
        g_A = self.audio_positional_encoding(g_A)
        channel_indices = torch.zeros(seq_length, dtype=torch.long).to(g_A.device)
        channel_embeddings = self.channel_embedding(channel_indices)
        g_A += channel_embeddings

        # 输入Transformer解码器
        g_A = g_A.unsqueeze(1)
        d_A = self.transformer_decoder(g_A)
        d_A = d_A.squeeze(1)
        return d_A


def predict_masked_audio_tokens(d_A, num_masked_tokens, output_dim):
    # 取与掩码音频令牌对应的音频特征子集
    masked_d_A = d_A[-num_masked_tokens:, :]

    # 上采样
    linear_layer = nn.Linear(masked_d_A.size(1), output_dim)
    upsampled_d_A = linear_layer(masked_d_A)

    # 重塑得到掩码令牌估计
    predicted_AM = upsampled_d_A.view(-1)
    return predicted_AM