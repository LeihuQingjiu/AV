# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.init import xavier_uniform_, constant_, normal_
from torch.nn import Parameter

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


def add_masked_embeddings(gAV, video_token_num, unmask_indices_list, mask_indices_list, mask_embeddings, device):
    """
    Args:
        gAV: [B, V+Q-S_min, D] 共享编码器输出
        video_token_num: 视频token数量 V
        unmask_indices_list: list([num_unmask]) 每个样本的unmask位置索引
        mask_indices_list: list([num_mask]) 每个样本的mask位置索引
        mask_embeddings: [S_total, D] 可学习的掩码嵌入
    Returns:
        full_gAV: [B, V+Q, D] 重组后的完整特征
    """
    batch_size = gAV.size(0)
    total_audio_tokens = video_token_num + max([len(unmask)+len(mask) for unmask, mask in zip(unmask_indices_list, mask_indices_list)])
    
    # 初始化全量特征矩阵
    full_gAV = torch.zeros(batch_size, total_audio_tokens, gAV.size(-1), device=device)
    
    # 视频部分直接填充
    full_gAV[:, :video_token_num] = gAV[:, :video_token_num]
    
    # 当前掩码嵌入的指针
    mask_ptr = 0
    
    for i in range(batch_size):
        # 当前样本的索引信息
        video_end = video_token_num
        audio_unmask_indices = unmask_indices_list[i] + video_end  # 转换为全局索引
        audio_mask_indices = mask_indices_list[i] + video_end
        
        # 获取当前样本的掩码数量
        num_mask = len(mask_indices_list[i])
        
        # 填充unmask部分
        full_gAV[i, audio_unmask_indices] = gAV[i, video_token_num:video_token_num+len(unmask_indices_list[i])].float()
        
        # 填充mask嵌入
        if num_mask > 0:
            # 获取当前样本对应的掩码嵌入
            sample_mask_emb = mask_embeddings[mask_ptr:mask_ptr+num_mask]  # [num_mask, D]
            full_gAV[i, audio_mask_indices] = sample_mask_emb
            mask_ptr += num_mask
            
    return full_gAV


class SharedAVDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, in_features):
        super(SharedAVDecoder, self).__init__()
        self.linear_projection = nn.Linear(in_features, out_features=384)
        self.masked_token_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hidden_dim)
        self.visual_positional_encoding = VisualPositionEncoding3D(hidden_dim)
        self.audio_positional_encoding = AudioPositionEncoding2D(hidden_dim)
        self.channel_embedding = nn.Embedding(2, hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=6, dim_feedforward=4 * hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        

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

    def forward(self, f_AV, video_token_num, unmask_indices_list, mask_indices_list, mask_emb):
        # 创建低维投影
        g_AV = self.linear_projection(f_AV)  # (batch_size, seq_length, hidden_dim)
        
        # masked_token_embeddings = self.masked_token_embedding(torch.zeros(S, dtype=torch.long).to(g_AV.device))  # (S, hidden_dim)
        # masked_token_embeddings = masked_token_embeddings.unsqueeze(0).expand(g_AV.size(0), -1, -1)  # (batch_size, S, hidden_dim)
        # g_AV = torch.cat((g_AV, masked_token_embeddings), dim=1)  # (batch_size, seq_length + S, hidden_dim)
        g_AV = add_masked_embeddings(
            g_AV, 
            video_token_num = video_token_num, 
            unmask_indices_list = unmask_indices_list, 
            mask_indices_list = mask_indices_list, 
            mask_embeddings = mask_emb, 
            device = g_AV.device
        )


        # 添加位置嵌入、通道嵌入和模态嵌入
        batch_size, seq_length, _ = g_AV.size()
        visual_seq_length = 330
        audio_seq_length = 784

        visual_part = g_AV[:, :visual_seq_length, :]  # (batch_size, visual_seq_length, hidden_dim)
        audio_part = g_AV[:, visual_seq_length:visual_seq_length + audio_seq_length, :]  # (batch_size, audio_seq_length, hidden_dim)

        visual_part = self.visual_positional_encoding(visual_part)  # (batch_size, visual_seq_length, hidden_dim)
        audio_part = self.audio_positional_encoding(audio_part)  # (batch_size, audio_seq_length, hidden_dim)

        left_channel_indices = torch.zeros(audio_seq_length // 2, dtype=torch.long).to(g_AV.device)  # (seq_length // 2,)
        right_channel_indices = torch.ones(audio_seq_length // 2, dtype=torch.long).to(g_AV.device)  # (seq_length // 2,)
        channel_indices = torch.cat((left_channel_indices, right_channel_indices), dim=0)  # (seq_length,)
        channel_embeddings = self.channel_embedding(channel_indices)  # (seq_length, hidden_dim)
        channel_embeddings = channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, hidden_dim)
        audio_part += channel_embeddings

        # channel_indices = torch.zeros(audio_seq_length, dtype=torch.long).to(audio_part.device)  # (audio_seq_length,)
        # channel_embeddings = self.channel_embedding(channel_info)  # (audio_seq_length, hidden_dim)
        # channel_embeddings = channel_embeddings.expand(batch_size, -1, -1)  # (batch_size, audio_seq_length, hidden_dim)
        # audio_part += channel_embeddings

        modality_indices_visual = torch.zeros(visual_seq_length, dtype=torch.long).to(visual_part.device)  # (visual_seq_length,)
        modality_indices_audio = torch.ones(audio_seq_length, dtype=torch.long).to(audio_part.device)  # (audio_seq_length,)
        modality_indices = torch.cat((modality_indices_visual, modality_indices_audio), dim=0)  # (visual_seq_length + audio_seq_length,)
        modality_embeddings = self.modality_embedding(modality_indices)  # (visual_seq_length + audio_seq_length, hidden_dim)
        modality_embeddings = modality_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, visual_seq_length + audio_seq_length, hidden_dim)

        g_AV = torch.cat((visual_part, audio_part), dim=1)  # (batch_size, seq_length + S, hidden_dim)
        g_AV += modality_embeddings

        # 输入浅层 Transformer 解码器
        g_AV = g_AV.transpose(0, 1)  # (seq_length + S, batch_size, hidden_dim)
        h_AV = self.transformer_decoder(g_AV, g_AV)  # (seq_length + S, batch_size, hidden_dim)
        h_AV = h_AV.transpose(0, 1)  # (batch_size, seq_length + S, hidden_dim)
        h_AV = self.layer_norm(h_AV)

        # 提取音频特征
        h_A = h_AV[:, visual_seq_length:, :]  # (batch_size, audio_seq_length, hidden_dim)
        return h_A


class AudioDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(AudioDecoder, self).__init__()
        self.audio_positional_encoding = AudioPositionEncoding2D(hidden_dim)
        self.channel_embedding = nn.Embedding(2, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

    def forward(self, g_A):
    # 重新添加位置嵌入和通道嵌入
        batch_size, seq_length, _ = g_A.size()
        g_A = self.audio_positional_encoding(g_A)  # (batch_size, seq_length, hidden_dim)

        # 添加通道嵌入
        left_channel_indices = torch.zeros(seq_length // 2, dtype=torch.long).to(g_A.device)  # (seq_length // 2,)
        right_channel_indices = torch.ones(seq_length // 2, dtype=torch.long).to(g_A.device)  # (seq_length // 2,)
        channel_indices = torch.cat((left_channel_indices, right_channel_indices), dim=0)  # (seq_length,)
        channel_embeddings = self.channel_embedding(channel_indices)  # (seq_length, hidden_dim)
        channel_embeddings = channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, hidden_dim)
        g_A += channel_embeddings
        # channel_indices = torch.zeros(seq_length, dtype=torch.long).to(g_A.device)  # (seq_length,)
        # channel_embeddings = self.channel_embedding(channel_indices)  # (seq_length, hidden_dim)
        # channel_embeddings = channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, hidden_dim)
        # g_A += channel_embeddings

        # 输入 Transformer 解码器
        g_A = g_A.transpose(0, 1)  # (seq_length, batch_size, hidden_dim)
        d_A = self.transformer_decoder(g_A, g_A)  # (seq_length, batch_size, hidden_dim)
        d_A = d_A.transpose(0, 1)  # (batch_size, seq_length, hidden_dim)
        return d_A

    
    
# class Predict_Masked_Audio_Tokens(nn.Module):
#     def __init__(self, in_features):
#         super(Predict_Masked_Audio_Tokens, self).__init__()
#         self.linear = nn.Linear(in_features, 2 * 16)  # 输入特征到补丁维度

#     def forward(self, d_A, mask_indices_list):
#         # import pdb; pdb.set_trace() 
#         batch_size, total_tokens,_ = d_A.shape
#         # batch_size, total_tokens, _ = mask.size(0), mask.size(1), d_A.size(2)
#         # masked_indices = torch.nonzero(mask)
#         masked_d_A = d_A[mask_indices_list[:, 0], mask_indices_list[:, 1], :]
#         upsampled_d_A = self.linear(masked_d_A)  
        
#         predicted_audio = torch.zeros(batch_size, total_tokens, 32, 
#                                     device=upsampled_d_A.device)
        
#         predicted_audio[mask_indices_list[:, 0], mask_indices_list[:, 1]] = upsampled_d_A.float()
#         print(f"Predicted audio shape: {predicted_audio.shape}")  # 应为 [1, 784, 32]

#         # final_audio = unmasked_audio + predicted_audio    # 保留未掩码的原始值
#         return predicted_audio

class Predict_Masked_Audio_Tokens(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * 16)
        self._init_weights()
        
    def _init_weights(self):
        init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0)

    def forward(self, d_A, masked_indices_list):
        batch_size, Q, _ = d_A.shape
        
        # 生成全局索引 [S_total, 2]
        global_indices = []
        for i in range(batch_size):
            local_indices = masked_indices_list[i]
            if local_indices.numel() == 0:
                continue
            batch_idx = torch.full_like(local_indices, i)
            global_batch_indices = torch.stack([batch_idx, local_indices], dim=1)
            global_indices.append(global_batch_indices)
        
        if not global_indices:
            return torch.zeros(batch_size, Q, 32, device=d_A.device, dtype=d_A.dtype)
        
        global_indices = torch.cat(global_indices, dim=0)
        
        # 提取被掩码的特征 [S_total, D]
        masked_d_A = d_A[global_indices[:, 0], global_indices[:, 1], :]
        
        # 通过线性层上采样 [S_total, 32]
        upsampled_d_A = self.linear(masked_d_A)  # 类型继承自 masked_d_A
        
        # Ensure the data types match
        upsampled_d_A = upsampled_d_A.to(d_A.dtype)
        
        predicted_audio = torch.zeros(
            batch_size, Q, 32, 
            device=d_A.device, 
            dtype=d_A.dtype 
        )
        predicted_audio[global_indices[:, 0], global_indices[:, 1]] = upsampled_d_A
        
        return predicted_audio