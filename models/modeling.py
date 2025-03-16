import torch
import torch.nn as nn

from models.Decoder import AudioDecoder, SharedAVDecoder, Predict_Masked_Audio_Tokens
from models.Encoder import AudioEncoder, VisualEncoder, SharedAVEncoder


class AVCorrModel(nn.Module):
    def __init__(self, config):
        super(AVCorrModel, self).__init__()
        # 编码器部分
        self.Encoder_hidden_dim = config['model']['Encoder_hidden_dim']
        self.Decoder_hidden_dim = config['model']['Decoder_hidden_dim']
        self.visual_encoder_layers = config['model']['visual_encoder_layers']
        self.audio_encoder_layers = config['model']['audio_encoder_layers']
        self.shared_encoder_layers = config['model']['shared_encoder_layers']
        self.shared_decoder_layers = config['model']['shared_decoder_layers']
        self.audio_decoder_layers = config['model']['audio_decoder_layers']
        self.audio_feat_dim = config['model']['audio_feat_dim']
        self.visual_feat_dim = config['model']['video_feat_dim']
        self.mask_embedding = nn.Parameter(torch.randn(self.Decoder_hidden_dim))
        self.channel_embed = nn.Embedding(num_embeddings=2, embedding_dim=32)

        self.visual_encoder = VisualEncoder(
            input_dim = 3 * 5 * 16 * 16,  # RGB * 5帧 * 16x16
            hidden_dim = config['model']['Encoder_hidden_dim'],
            num_layers = config['model']['visual_encoder_layers']
        )

        self.audio_encoder = AudioEncoder(
            input_dim = 2*16,  # 双通道 * 16时间步
            hidden_dim = config['model']['Encoder_hidden_dim'],
            num_layers = config['model']['audio_encoder_layers']
        )

        # 共享编码器
        self.shared_encoder = SharedAVEncoder(
            hidden_dim = config['model']['Encoder_hidden_dim'],
            num_layers = config['model']['shared_encoder_layers'],
            visual_feature_dim = config['model']['video_feat_dim'],
            audio_feature_dim = config['model']['audio_feat_dim']
        )

        # 解码器部分
        self.shared_decoder = SharedAVDecoder(
            hidden_dim = config['model']['Decoder_hidden_dim'],
            num_layers = config['model']['shared_decoder_layers'],
            in_features= config['model']['shared_feat_dim']
            
        )
        
        self.audio_decoder = AudioDecoder(  
            hidden_dim = config['model']['Decoder_hidden_dim'],
            num_layers = config['model']['audio_decoder_layers']
        )

        self.predict_masked_audio_tokens = Predict_Masked_Audio_Tokens(
            in_features = config['model']['Decoder_hidden_dim']
            # output_dim = 32  # config['model']['audio_feat_dim']
        )
        


    def _generate_mixed_mask(self, audio, mask_ratio, num_mask=None):
        """
        Args:
            audio: [B, T, C] (e.g., [16, 784, 32])
            mask_ratio: r 的概率使用全通道掩码
            num_mask: 部分掩码的数量（可选）
        Returns:
            mask: [B, T] (True 表示被掩码的位置)
        """
        B, T, C = audio.shape
        device = audio.device
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        
        # 每个样本独立决定是否全通道掩码
        is_full_mask = torch.rand(B, device=device) < mask_ratio  # [B]
        
        # --- 全通道掩码分支 ---
        full_mask_indices = torch.where(is_full_mask)[0]  # 需要全掩码的样本下标
        if len(full_mask_indices) > 0:
            is_left = torch.randint(0, 2, (len(full_mask_indices),), device=device).bool()
            half_T = T // 2
            for i, idx in enumerate(full_mask_indices):
                if is_left[i]:
                    mask[idx, :half_T] = True  # 掩码左半通道
                else:
                    mask[idx, half_T:] = True  # 掩码右半通道
        
        # --- 部分随机掩码分支 ---
        partial_mask_indices = torch.where(~is_full_mask)[0]  # 需要部分掩码的样本下标
        if len(partial_mask_indices) > 0:
            # 计算部分掩码的数量（默认 20%）
            S = int(T * 0.2) if num_mask is None else num_mask
            for idx in partial_mask_indices:
                # 随机选择 S 个位置掩码
                positions = torch.randperm(T, device=device)[:S]
                mask[idx, positions] = True
        
        return mask  # [B, 784]

    
    def _extract_unmask_audio(self, mask, audio, device):
        """_summary_

        Args:
            mask (_type_): _description_
            audio (_type_): _description_
        """
        batch_size = mask.shape[0]
        
        masked_audio_list = []  
        unmask_audio_list = []    
        masked_indices_list = []   # mask位置的原始索引
        unmask_indices_list = []  # unmask位置的原始索引

        for i in range(batch_size):
            # --- 提取mask部分 ---
            sample_mask = mask[i]  # [784]
            sample_audio = audio[i]  # [784, 32]
            
            # mask位置的索引（mask=1的位置）
            masked_indices = torch.where(sample_mask)[0]  # [num_mask]
            masked_audio = sample_audio[masked_indices]    # [num_mask, 32]
            masked_audio_list.append(masked_audio)
            masked_indices_list.append(masked_indices)
            
            # --- 提取unmask部分 ---
            unmask_mask = ~sample_mask
            unmask_indices = torch.where(unmask_mask)[0]  # [num_unmask]
            unmask_audio = sample_audio[unmask_indices]   # [num_unmask, 32]
            unmask_audio_list.append(unmask_audio)
            unmask_indices_list.append(unmask_indices)

        # 动态填充mask
        max_mask_len = max([t.size(0) for t in masked_audio_list])
        padded_mask = torch.zeros(batch_size, max_mask_len, 32)
        mask_mask = torch.zeros(batch_size, max_mask_len, dtype=torch.bool)
        for i, tokens in enumerate(masked_audio_list):
            num_mask = tokens.size(0)
            padded_mask[i, :num_mask] = tokens
            mask_mask[i, :num_mask] = True

        # 动态unmasked
        max_unmask_len = max([t.size(0) for t in unmask_audio_list])
        padded_unmask = torch.zeros(batch_size, max_unmask_len, 32, device=device)
        unmask_mask = torch.zeros(batch_size, max_unmask_len, dtype=torch.bool, device=device)
        for i, tokens in enumerate(unmask_audio_list):
            num_unmask = tokens.size(0)
            padded_unmask[i, :num_unmask] = tokens
            unmask_mask[i, :num_unmask] = True
            
        return padded_unmask, unmask_mask, unmask_indices_list, masked_indices_list
    
    

    def forward(self, video, audio, mask_ratio=0.2):
        device = video.device  # Ensure all tensors are on the same device
        audio = audio.to(device)

        
        # 生成混合掩码
        mask = self._generate_mixed_mask(audio, mask_ratio) # 掩码张量 
        
        unmask_audio, unmask_mask, unmask_indices_list, mask_indices_list = self._extract_unmask_audio(mask, audio, device)
        tokens = audio.shape[1]        
        unmask_audio = unmask_audio.to(device)

        # 编码阶段
        visual_feats = self.visual_encoder(video)
        audio_feats = self.audio_encoder(unmask_audio, unmask_mask, unmask_indices_list, tokens)  
        shared_feats = torch.cat([visual_feats, audio_feats], dim=1)  # [B, V+max_S, D]
        print("编码完成")
        
        # 解码阶段
        S_total = sum(len(mask_indices) for mask_indices in mask_indices_list)
        mask_emb = self.mask_embedding.unsqueeze(0).expand(S_total, -1)
        h_A = self.shared_decoder(shared_feats, visual_feats.shape[1], unmask_indices_list, mask_indices_list, mask_emb)
        d_A = self.audio_decoder(h_A)
        print(f"d_A shape: {d_A.shape}")
        pred_audio = self.predict_masked_audio_tokens(d_A, mask_indices_list)
        
        print(f"pred_audio shape: {pred_audio.shape}")

        return pred_audio, mask 
    
    
