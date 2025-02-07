import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torchaudio
import torchvision.transforms as transforms
from torch.nn.init import xavier_uniform_, constant_, normal_
from Encoder import VisualEncoder, AudioEncoder, SharedAVEncoder
from Decoder import SharedAVDecoder, AudioDecoder, predict_masked_audio_tokens
from torch.distributions.truncated_normal import TruncatedNormal


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.x_dim = 36
        self.y_dim = 128
        self.output_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_encoder = AudioEncoder(input_dim=2 * 16, hidden_dim=self.hidden_dim, num_layers=2)
        self.visual_encoder = VisualEncoder(input_dim=330 * 16 * 16, hidden_dim=self.hidden_dim, num_layers=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                if hasattr(m, 'modality_embedding') or hasattr(m, 'channel_embedding'):
                    # 可学习的模态和通道嵌入令牌
                    trunc_normal = TruncatedNormal(-2, 2, 0, 0.02)
                    m.weight.data = trunc_normal.sample(m.weight.data.shape).to(self.device)
                elif hasattr(m, 'mask_token_embedding'):
                    # 掩码令牌
                    normal_(m.weight, std=0.02)

    def forward(self, visual_tokens, audio_tokens):
        r = 0.2
        masked_audio, unmasked_audio = self.mask_audio(audio_tokens, r)

        # 对video和audio进行编码
        visual_tokens = visual_tokens.view(visual_tokens.size(0), -1)  # tokens调整形状
        visual_features = self.visual_encoder(visual_tokens)

        unmasked_audio = unmasked_audio.view(unmasked_audio.size(0), -1)
        audio_features = self.audio_encoder(unmasked_audio)

        # ShareEncoder
        av_features = SharedAVEncoder(visual_features, audio_features)

        # decoder
        shared_av_decoder = SharedAVDecoder(256, 128, 2)
        audio_decoder = AudioDecoder(256, 128, 2)

        h_A = shared_av_decoder(av_features)
        d_A = audio_decoder(h_A)

        num_masked_tokens = 1
        output_dim = 32
        predicted_AM = predict_masked_audio_tokens(d_A, num_masked_tokens, output_dim)

        return predicted_AM

    def mask_audio(self, audio, r):
        batch_size, channels, height, width = audio.shape
        mask = torch.ones_like(audio, device=self.device)
        for i in range(batch_size):
            x = np.random.uniform(0, 1)
            if x <= r:
                # 随机掩码整个音频通道
                channel_to_mask = np.random.randint(0, 2)
                mask[i, channel_to_mask, :, :] = 0
            else:
                # 随机掩码音频令牌
                num_tokens_to_mask = int(0.2 * height * width)  # 这里假设掩码20%的令牌，可根据实际调整
                indices = np.random.choice(height * width, num_tokens_to_mask, replace=False)
                rows = indices // width
                cols = indices % width
                mask[i, :, rows, cols] = 0

        # 计算掩码和未掩码部分
        masked_audio = audio * mask
        unmasked_audio = audio * (1 - mask)

        return masked_audio, unmasked_audio


def dataprocessing(video_paths, audio_paths):
    visual_tokens = []
    audio_tokens = []
    for video_path, audio_path in zip(video_paths, audio_paths):
        # 读取视频
        video, _, _ = read_video(video_path)  # video (T, H, W, C)

        # ImageNet的均值和标准差
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

        # 转换为torch张量并归一化
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        video = normalize(video)

        # 生成视觉tokens
        patch_size = 16
        tubelet_length = 5
        num_tubelets = 330
        tubelet_step = (video.shape[2] * video.shape[3]) // num_tubelets
        for i in range(num_tubelets):
            start = i * tubelet_step
            end = start + tubelet_length * patch_size * patch_size
            tubelet = video.new_zeros((tubelet_length, 3, patch_size, patch_size))
            for j in range(tubelet_length):
                patch_start = start + j * patch_size * patch_size
                patch_end = patch_start + patch_size * patch_size
                patch = video[:, :, patch_start // video.shape[3]:(patch_start // video.shape[3]) + patch_size,
                              (patch_start % video.shape[3]):(patch_start % video.shape[3]) + patch_size]
                tubelet[j] = patch
            visual_tokens.append(tubelet)

        # 读取音频
        audio, sr = torchaudio.load(audio_path)
        # 归一化音频
        audio = audio / torch.max(torch.abs(audio))
        audio = torch.clamp(audio, -1, 1)

        # 计算梅尔频谱图
        spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, win_length=int(0.025 * sr),
                                                                     hop_length=int(0.01 * sr), n_mels=128)
        spectrogram = spectrogram_transform(audio)

        # 归一化频谱图
        spectrogram_mean = spectrogram.mean(dim=(0, 2, 3), keepdim=True)
        spectrogram_std = spectrogram.std(dim=(0, 2, 3), keepdim=True)
        spectrogram = (spectrogram - spectrogram_mean) / spectrogram_std

        # 生成音频tokens
        audio_token_size = (2, 16)
        num_audio_tokens = 392
        for i in range(num_audio_tokens):
            start_x = i * audio_token_size[1]
            end_x = start_x + audio_token_size[1]
            audio_token = spectrogram[:, :, :, start_x:end_x]
            audio_tokens.append(audio_token)

    visual_tokens = torch.stack(visual_tokens)
    audio_tokens = torch.stack(audio_tokens)
    return visual_tokens, audio_tokens


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (video_paths, audio_paths, masked_audio_tokens) in enumerate(train_loader):
        visual_tokens, audio_tokens = dataprocessing(video_paths, audio_paths)
        visual_tokens, audio_tokens, masked_audio_tokens = visual_tokens.to(device), audio_tokens.to(device), masked_audio_tokens.to(device)

        optimizer.zero_grad()
        outputs = model(visual_tokens, audio_tokens)
        loss = criterion(outputs, masked_audio_tokens)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)

    # 定义损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=10),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 - 10)
        ],
        milestones=[10]
    )

    #
    video_paths = []
    audio_paths = []
    masked_audio_tokens = []
    # 这里可以根据实际情况从数据集读取
    dataset = list(zip(video_paths, audio_paths, masked_audio_tokens))
    train_loader = DataLoader(dataset, batch_size=104, shuffle=True)

    num_epochs = 200
    for epoch in range(num_epochs):
        epoch_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
