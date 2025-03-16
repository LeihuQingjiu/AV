import os
import decord
import torchaudio
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize


# ImageNet 均值和标准差
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

GLOBAL_MEAN = -7.523139
GLOBAL_STD = 3.413932

# GLOBAL_MEAN = -7.152682
# GLOBAL_STD = 3.544300

import matplotlib.pyplot as plt
def visualize_spectrogram(mel_spec, title="Mel Spectrogram", save_path=None):
    """绘制梅尔频谱图"""
    plt.figure(figsize=(12, 4))
    plt.imshow(mel_spec, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Bands')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_patches(audio_patches, channel=0, save_path=None):
    plt.figure(figsize=(12, 8))
    # 输入形状: (8, 49, 32) → 转换为 (8, 49, 16, 2)
    patches = audio_patches.view(8, 49, 16, 2).detach().cpu().numpy()
    full_image = np.zeros((128, 98))
    
    for mel_idx in range(8):
        for time_idx in range(49):
            y_start = mel_idx * 16
            x_start = time_idx * 2
            full_image[y_start:y_start+16, x_start:x_start+2] = patches[mel_idx, time_idx]
    
    plt.imshow(full_image, origin='lower', aspect='auto', cmap='viridis')
    plt.title("Reconstructed Spectrogram from Patches")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        # dataset_root = config['data']['root']
        self.root = os.path.join(config['data']['root_dir'], split)
        # dataset_root = "data/egocom_test"
        self.samples = self._find_videos(self.root)
        print(f"Loaded {len(self.samples)} video samples")

        # 视频参数
        self.target_fps = 5
        self.tubelet_t = config['data']['tubelet_size'][0]
        self.tubelet_h = config['data']['tubelet_size'][1]
        self.tubelet_w = config['data']['tubelet_size'][2]

        # 音频参数
        self.sample_rate = config['data']['audio_spec']['sample_rate']  # 16000  # 16kHz
        self.n_fft = config['data']['audio_spec']['n_fft']  # 512  32ms窗口
        self.n_mels = config['data']['audio_spec']['n_mels']  # 128个梅尔频率
        self.hop_length = config['data']['audio_spec']['hop_length']  # 16kHz下10ms的hop长度

        # 25ms 窗口长度对应的样本数
        self.window_length = int(0.025 * self.sample_rate)
        self.resize = Resize((240, 352), antialias=True)        # 对视频下采样
        
    def _find_videos(self, root):
        root = Path(root)
        return list(root.glob('*.[mM][pP][4vV]'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            try:
                video_path = self.samples[idx]

                # ========== 视频处理 ==========
                # 使用decord以5fps读取视频
                vr = decord.VideoReader(str(video_path))
                original_fps = vr.get_avg_fps()

                # 计算需要采样的帧索引
                num_frames = len(vr)
                frame_indices = self._sample_frames(num_frames, original_fps)

                # 读取并预处理视频帧
                video = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
                
                video = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W)
                video = self.resize(video)
                video = video / 255.0 # 归一化到 [0, 1]
                
                # 使用 ImageNet 均值和标准差归一化
                video = (video - IMAGENET_MEAN) / IMAGENET_STD

                # 确保视频长度为1秒（5帧）
                if video.shape[0] < 5:
                    video = self._pad_video(video)
                else:
                    video = video[:5]

                # ========== 音频处理 ==========
                # 使用torchaudio直接读取音频
                waveform, sr = torchaudio.load(video_path)
                

                # 选择前两个声道作为双耳音频
                if waveform.shape[0] > 2:
                    waveform = waveform[:2]

                # 重采样到16kHz
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

                # 切割/填充至1秒长度
                target_samples = self.sample_rate  # 16000 samples
                if waveform.shape[1] < target_samples:
                    waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target_samples]

                # 归一化到 [-1, 1]
                waveform = torch.clamp(waveform, -1, 1)
                
                with torch.no_grad():
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_fft=self.n_fft,
                        n_mels=self.n_mels,
                        hop_length=self.hop_length,
                        win_length=self.window_length
                    )(waveform)  # (2, n_mels, time_steps)

                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))
                print(f"Mel spec range: {mel_spec.min().item():.2f} ~ {mel_spec.max().item():.2f}")
                mel_spec = (mel_spec - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
                
                # 插值 
                if mel_spec.shape[-1] != 98:
                    print(f"原始形状: {mel_spec.shape}")  # 调试日志
                    mel_spec = F.interpolate(
                        mel_spec.unsqueeze(0),
                        size=(128, 98),  
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    print(f"调整后形状: {mel_spec.shape}")


                # ========== 分块处理 ==========
                video_tokens = self._process_video(video)
                audio_tokens = self._process_audio(mel_spec)
                print(f"Video tokens shape: {video_tokens.shape}, Audio tokens shape: {audio_tokens.shape}")
                
                
                # 绘制梅尔频谱图：
                if idx == 0:
                    print(111111111111111111111111111111111111)
                    original_spec = torch.exp(mel_spec * GLOBAL_STD + GLOBAL_MEAN)[0].detach()
                    # original_spec = torch.exp(mel_spec * GLOBAL_STD + GLOBAL_MEAN)
                    visualize_spectrogram(
                        original_spec, 
                        title="Original (Channel 0)", 
                        save_path="spectrogram.png"
                    )
                    patched_spec = audio_tokens.view(2, 8, 49, 32)[0].detach()
                    visualize_patches(
                        patched_spec, 
                        save_path="patches.png"
                    )
                
                
                assert torch.isfinite(video).all(), "视频数据包含NaN或Inf"
                assert torch.isfinite(mel_spec).all(), "梅尔频谱包含NaN或Inf"
                assert mel_spec.min() > -1e5 and mel_spec.max() < 1e5, "梅尔频谱数值异常"
                return {
                    'video': video_tokens,
                    'audio': audio_tokens
                }
            except Exception as e:
                # 处理首次尝试加载视频就出错的情况
                if 'video_path' not in locals():
                    video_path = "unknown"
                print(f"Error loading {video_path}: {e}")
                idx = (idx + 1) % len(self.samples)

    def _sample_frames(self, total_frames, original_fps):
        """采样5帧/秒的视频帧"""
        target_count = 5  # 1秒5帧
        step = max(1, int(original_fps / self.target_fps))
        indices = list(range(0, total_frames, step))[:target_count]

        # 不足时重复最后一帧
        if len(indices) < target_count:
            indices += [indices[-1]] * (target_count - len(indices))
        return indices

    def _pad_video(self, video):
        """填充视频到5帧"""
        pad_size = 5 - video.shape[0]
        # 完成填充操作并返回填充后的视频张量
        return torch.cat([video, video[-1:].repeat(pad_size, 1, 1, 1)], dim=0)

    def _process_video(self, video):
        """视频分块处理"""
        # video shape: (T=5, C, H, W)
        T, C, H, W = video.shape
        assert H == 240 and W == 352, f"视频尺寸必须为 240x352, 当前尺寸: {H}x{W}"
        num_h = H // self.tubelet_h  # 240 / 16 = 15
        num_w = W // self.tubelet_w  # 352 / 16 = 22
        video = video[:, :, :num_h*self.tubelet_h, :num_w*self.tubelet_w]  # 裁剪到可整除尺寸
        
        tubelets = video.unfold(2, self.tubelet_h, self.tubelet_h).unfold(3, self.tubelet_w, self.tubelet_w)
        tubelets = tubelets.permute(2, 3, 0, 1, 4, 5).reshape(-1, T, C, self.tubelet_h, self.tubelet_w)
        video_tokens = tubelets.reshape(tubelets.size(0), -1)  # (num_tokens, token_dim)
    
        # 分割成 P 个不重叠的 tubelets
        # tubelets = []
        # for i in range(0, H, self.tubelet_h):
        #     for j in range(0, W, self.tubelet_w):
        #         # 避免越界
        #         tubelet = video[:, :, i:min(i+self.tubelet_h, H), j:min(j+self.tubelet_w, W)]
        #         if tubelet.shape[2] == self.tubelet_h and tubelet.shape[3] == self.tubelet_w:
        #             tubelets.append(tubelet)
        # video_tokens = torch.stack(tubelets)
        # video_tokens = video_tokens.reshape(-1, T * C * self.tubelet_h * self.tubelet_w)
        return video_tokens

    def _process_audio(self, audio):
        """音频分块处理"""
        # audio shape: (2, n_mels=128, time_steps=98)
        # channels, n_mels, time_steps = audio.shape
        # assert n_mels % 16 == 0 and time_steps % 2 == 0, "n_mels must be divisible by 16 and time_steps must be divisible by 2"
        # audio = audio.unfold(2, 2, 2).unfold(1, 16, 16)  # (2, n_mels//16, time_steps//2, 2, 16)  2*16
        # print(f"n_mels//16: {n_mels//16}, time_steps//2: {time_steps//2}")
        # all_tokens = []
        # for channel in range(channels):
        #     channel_tokens = audio[channel].contiguous().view(-1, 2 * 16)  # (392, 2 * 16)
        #     all_tokens.append(channel_tokens)
        # # 将列表转换为张量
        # all_tokens = torch.cat(all_tokens, dim=0)
        assert audio.shape[1] == 128 and audio.shape[2] == 98, f"音频形状必须为 (2,128,98), 当前形状: {audio.shape}"
    
    # 分块逻辑（固定形状）
        # audio = audio.unfold(1, 16, 16).unfold(2, 2, 2)  # (2, 8, 49, 16, 2) 分别在梅尔与时间维度分块
        # audio = audio.permute(0, 1, 2, 4, 3).reshape(2, 8 * 49, 32)
        # audio_tokens = audio.reshape(-1, 32)  # (2 * 8 * 49=784, 32)
        
        channels, n_mels, time_steps = audio.shape
        assert n_mels % 16 == 0, "梅尔频带数必须能被16整除"
        assert time_steps % 2 == 0, "时间步数必须能被2整除"
        
        # 分块操作
        audio = audio.unfold(1, 16, 16)    # 在梅尔维度分块 (2, 8, 98, 16)
        audio = audio.unfold(2, 2, 2)      # 在时间维度分块 (2, 8, 49, 2, 16)
        # 维度重组 (2, 8, 49, 32)
        audio = audio.permute(0, 1, 2, 4, 3).contiguous()
        # audio = audio.view(channels, 8, 49, 32)
        audio = audio.view(2, 8, 49, 32)
        
        # 合并所有维度 (batch_size, tokens_num, token_dim)
        audio_tokens = audio.view(-1, 32)  # (2 * 8 * 49=784, 32)
        # return all_tokens
        return audio_tokens

    @staticmethod
    def _reconstruct_mel(tokens):
        """修正版分块重构方法"""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # 添加batch维度
            
        batch_size = tokens.shape[0]
        
        # 重构为原始分块形状
        reconstructed = tokens.view(batch_size, 2, 8, 49, 32)
        
        # 分解频带和时间维度
        reconstructed = reconstructed.view(batch_size, 2, 8, 49, 16, 2)
        
        # 逆分块操作
        reconstructed = reconstructed.permute(0, 1, 2, 5, 3, 4).contiguous()
        reconstructed = reconstructed.view(batch_size, 2, 128, 98)  # (B, 2, 128, 98)
        
        return reconstructed.squeeze()
    
    
    
if __name__ == "__main__":
    from configs.configs import load_config
    config = load_config('configs/base.yaml')
    GLOBAL_MEAN, GLOBAL_STD = compute_global_stats(config)
    print(f"GLOBAL_MEAN = {GLOBAL_MEAN:.4f}, GLOBAL_STD = {GLOBAL_STD:.4f}")
    