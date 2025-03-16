import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import numpy as np
from configs.configs import load_config
import functools as F

def compute_global_stats(config_path, single_file_path=None):
    """支持计算单个音频文件的统计量"""
    # 加载配置
    config = load_config(config_path)
    audio_spec = config['data']['audio_spec']
    
    # 梅尔频谱参数
    sample_rate = audio_spec['sample_rate']
    n_fft = audio_spec['n_fft']
    n_mels = audio_spec['n_mels']
    hop_length = audio_spec['hop_length']
    window_length = int(0.025 * sample_rate)
    
    # 创建梅尔频谱转换器
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=window_length
    )
    
    # 处理单个文件模式
    if single_file_path:
        video_files = [Path(single_file_path)]
        print(f"Processing single file: {video_files[0].name}")
    else:
        # 默认处理训练集
        train_root = os.path.join(config['data']['root_dir'], 'train')
        video_files = list(Path(train_root).glob('*.[mM][pP][4vV]'))
        print(f"Found {len(video_files)} training videos")
    
    # 统计量计算
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    
    for video_path in tqdm(video_files, desc="Processing audio"):
        try:
            # 加载音频
            waveform, sr = torchaudio.load(video_path)
            
            # 处理声道
            if waveform.shape[0] > 2:
                waveform = waveform[:2]
            
            # 重采样
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
            # 标准化长度到1秒
            target_samples = sample_rate
            if waveform.shape[1] < target_samples:
                waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_samples]
            
            # 生成梅尔频谱
            mel_spec = mel_transform(waveform)
            log_mel = torch.log(mel_spec + 1e-8)
            
            # 累积统计量
            total_sum += log_mel.double().sum().item()
            total_sq += (log_mel.double() ** 2).sum().item()
            total_count += log_mel.numel()
            
        except Exception as e:
            print(f"\nSkipped {video_path.name}: {str(e)[:50]}")
            continue
    
    # 计算结果
    global_mean = total_sum / total_count
    global_std = np.sqrt(total_sq / total_count - global_mean ** 2)
    
    # 输出结果
    print("\n" + "=" * 40)
    if single_file_path:
        print(f"[单文件模式] {video_files[0].name} 统计量:")
    else:
        print(f"全局统计量 (基于 {len(video_files)} 个文件):")
    print(f"MEAN = {global_mean:.6f}")
    print(f"STD  = {global_std:.6f}")
    print("=" * 40)
    
    return global_mean, global_std

if __name__ == "__main__":
    # 使用方式1: 计算全局统计量 (默认)
    # compute_global_stats("configs/base.yaml")
    
    # 使用方式2: 计算单个文件统计量
    single_audio_path = "data/egocom_test/vid_001__day_1__con_1__person_1_part1.MP4"
    compute_global_stats("configs/base.yaml", single_file_path=single_audio_path)