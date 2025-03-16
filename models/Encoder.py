import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_


class SinPositionEncoding(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        batch_size, sequence_length, d_model = x.size()  # 正确获取维度顺序
        pe = torch.zeros(sequence_length, d_model, dtype=torch.float).to(x.device)
        exp_1 = torch.arange(d_model // 2, dtype=torch.float).to(x.device)
        exp_value = exp_1 / (d_model / 2)
        alpha = 1 / (self.base ** exp_value)
        out = torch.arange(sequence_length, dtype=torch.float).to(x.device)[:, None] @ alpha[None, :]
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)
        pe[:, 0::2] = embedding_sin
        pe[:, 1::2] = embedding_cos
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展至(batch_size, seq_len, d_model)
        return x + pe

class VisualPositionEncoding3D(nn.Module):
    """修正三维位置编码的维度合并方式"""
    def __init__(self, d_model, base=10000):
        super().__init__()
        assert d_model % 3 == 0, "d_model必须能被3整除"
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 动态计算三维分解
        grid_size = self.find_optimal_3d_factors(seq_len)
        H, W, T = grid_size
        assert H * W * T == seq_len, f"分解失败: {H}x{W}x{T} != {seq_len}"
        
        # 生成各维度编码
        dim_per_axis = self.d_model // 3
        device = x.device
        
        # 高度编码 (H维度)
        h_coord = torch.arange(H, device=device).float()
        pe_h = self._axis_encoding(h_coord, dim_per_axis)  # [H, dim_per_axis]
        
        # 宽度编码 (W维度)
        w_coord = torch.arange(W, device=device).float()
        pe_w = self._axis_encoding(w_coord, dim_per_axis)  # [W, dim_per_axis]
        
        # 时间编码 (T维度)
        t_coord = torch.arange(T, device=device).float()
        pe_t = self._axis_encoding(t_coord, dim_per_axis)  # [T, dim_per_axis]
        
        # 三维扩展并拼接
        pe = torch.cat([
            pe_h.unsqueeze(1).unsqueeze(2).expand(-1, W, T, -1),  # [H, W, T, D/3]
            pe_w.unsqueeze(0).unsqueeze(2).expand(H, -1, T, -1),  # [H, W, T, D/3]
            pe_t.unsqueeze(0).unsqueeze(1).expand(H, W, -1, -1)   # [H, W, T, D/3]
        ], dim=-1)  # [H, W, T, D]
        
        return x + pe.view(1, seq_len, self.d_model)

    def _axis_encoding(self, coord, dim_per_axis):
        """生成单轴位置编码"""
        div_term = torch.exp(
            torch.arange(0, dim_per_axis, 2, device=coord.device).float() *
            (-math.log(self.base) / dim_per_axis)
        )
        factors = coord.view(-1, 1) * div_term.view(1, -1)  # [L, D//2]
        
        pe = torch.zeros(len(coord), dim_per_axis, device=coord.device)
        pe[:, 0::2] = torch.sin(factors)
        pe[:, 1::2] = torch.cos(factors)
        return pe

    def find_optimal_3d_factors(self, n):
        """优化后的三维分解方法"""
        # 保持与论文一致的默认配置
        if n == 8 * 8 * 16:  # 1024
            return (8, 8, 16)
        
        # 寻找最接近论文配置的分解
        best_score = float('inf')
        best_factors = (1, 1, n)
        
        for h in self._get_factors(n):
            for w in self._get_factors(n//h):
                t = n // (h * w)
                if h * w * t != n: continue
                
                # 评分标准：空间接近8x8，时间接近16
                spatial_diff = abs(h*w - 64) + abs(h - w)*0.5
                temporal_diff = abs(t - 16)
                score = spatial_diff + temporal_diff * 3
                
                if score < best_score:
                    best_score = score
                    best_factors = (h, w, t)
        return best_factors

    def _get_factors(self, n):
        """获取n的所有因数对"""
        factors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n//i)
        return sorted(factors)

class AudioPositionEncoding2D(nn.Module):
    """2D位置编码（时间+频率）"""
    def __init__(self, d_model, time_steps=300, freq_bins=128, base=10000):
        super().__init__()
        self.d_model = d_model
        
        # 时间编码
        self.time_enc = SinPositionEncoding(d_model//2, base)
        # 频率编码
        self.freq_enc = SinPositionEncoding(d_model//2, base)
        
    def forward(self, x):
        # x shape: (batch, time_steps*freq_bins, d_model)
        batch_size, seq_len, _ = x.size()
        
        # 拆分时间和频率维度
        time_features = x[..., :self.d_model//2] + self.time_enc(x[..., :self.d_model//2])
        freq_features = x[..., self.d_model//2:] + self.freq_enc(x[..., self.d_model//2:])
        return torch.cat([time_features, freq_features], dim=-1)





class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=8):  # 论文指定8层
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = AudioPositionEncoding2D(hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=4*hidden_dim,
            batch_first=True,
            norm_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.channel_embed = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim)
        self._init_weights()


    def create_channel_labels(self, padded_unmask, unmask_indices_list, tokens, device):
        B, max_unmask_len = padded_unmask.shape[:2]
        channel_labels = torch.zeros((B, max_unmask_len), dtype=torch.long, device=device)
        
        for i in range(B):
            indices = unmask_indices_list[i]  # 获取样本i的原始位置索引
            labels = (indices >= (tokens // 2)).long()  # 左0，右1
            valid_len = len(indices)
            if valid_len > 0:
                channel_labels[i, :valid_len] = labels  # 填充有效标签
        return channel_labels

    def add_channel_embedding(self, padded_unmask, unmask_mask, unmask_indices_list, tokens):
        # 生成声道标签 [B, max_unmask_len]
        channel_labels = self.create_channel_labels(
            padded_unmask, unmask_indices_list, tokens, padded_unmask.device
        )
        
        # 获取声道嵌入 [B, max_unmask_len, 32]
        channel_embeddings = self.channel_embed(channel_labels)
        
        # 屏蔽填充位置的嵌入（无效位置置零）
        unmask_mask_3d = unmask_mask.unsqueeze(-1).float().to(padded_unmask.device)  # [B, max_unmask_len, 1]
        channel_embeddings = channel_embeddings * unmask_mask_3d
        
        # 将嵌入添加到未掩码音频
        return channel_embeddings

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                normal_(m.weight, std=0.02)
                
    def forward(self, unmasked_audio, unmask_mask, unmask_indices_list, tokens):
        x = self.proj(unmasked_audio)

        # 动态生成通道索引
        x = self.pos_encoder(x)  # [B, S, D]
        x += self.add_channel_embedding(unmasked_audio, unmask_mask, unmask_indices_list, tokens)
        return self.transformer_encoder(x)
    

class VisualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=8):  # 论文指定8层
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = VisualPositionEncoding3D(hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=4*hidden_dim,
            batch_first=True,
            norm_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x):
        """
        输入:
        x: 视觉特征 (batch, H×W×T, input_dim)
        """
        x = self.proj(x)
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)
    
    
class SharedAVEncoder(nn.Module):
    def __init__(self, visual_feature_dim, audio_feature_dim, hidden_dim, num_layers):
        super(SharedAVEncoder, self).__init__()
        assert visual_feature_dim == audio_feature_dim, "Visual and audio feature dimensions must match"
        self.visual_positional_encoding = SinPositionEncoding(visual_feature_dim)
        self.audio_positional_encoding = SinPositionEncoding(audio_feature_dim)
        self.channel_embedding = nn.Embedding(2, audio_feature_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=12, batch_first=True, 
            dim_feedforward=4 * hidden_dim,
            norm_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers)
        
        self.visual_modality_id = 0
        self.audio_modality_id = 1

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                if m.weight.requires_grad:
                    normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

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
        # channel_indices = torch.zeros(batch_size, audio_seq_length, dtype=torch.long).to(combined_features.device)
        left_len = audio_seq_length // 2
        device = visual_features.device
        channel_indices = torch.cat([
            torch.zeros(batch_size, left_len), 
            torch.ones(batch_size, audio_seq_length - left_len)
            ], dim=1).long().to(device)
        channel_embeddings = self.channel_embedding(channel_indices)
        combined_features[:, visual_seq_length:, :] += channel_embeddings

        # 为所有特征添加可学习的模态嵌入
        modality_indices_visual = torch.zeros(batch_size, visual_seq_length, dtype=torch.long).to(combined_features.device)
        modality_indices_audio = torch.ones(batch_size, audio_seq_length, dtype=torch.long).to(combined_features.device)
        modality_indices = torch.cat((modality_indices_visual, modality_indices_audio), dim=1)

        self.modality_mask = torch.cat([
            torch.full((visual_features.size(1),), self.visual_modality_id, device=visual_features.device),
            torch.full((audio_features.size(1),), self.audio_modality_id, device=audio_features.device)
        ], dim=0).unsqueeze(0)  # (1, S_v+S_a)
        
        self.visual_origin = visual_features
        self.audio_origin = audio_features
        
        modality_embeddings = self.modality_embedding(modality_indices)
        combined_features += modality_embeddings

        combined_features = combined_features.permute(1, 0, 2)
        av_features = self.transformer_encoder(combined_features)
        av_features = av_features.permute(1, 0, 2)
        print("av_features.shape:", av_features.shape)
        
        return av_features