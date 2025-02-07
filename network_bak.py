import torch
import pickle
import os, re
import numpy as np
import librosa
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.x_dim = 36
        self.y_dim = 128
        self.output_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.audiomask = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.x_dim * self.y_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.x_dim * self.y_dim),
            nn.Sigmoid()
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 9, self.hidden_dim)
        )

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, self.hidden_dim)
        )

        self.fc1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, audio, visual_input):
        mel_features = []
        # audio = np.array(audio)
        audio = audio.cpu().numpy()
        for i in range(audio.shape[0]):
            left_channel = audio[i, 0, :]
            right_channel = audio[i, 1, :]

            mel_left = librosa.feature.melspectrogram(y=left_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            mel_right = librosa.feature.melspectrogram(y=right_channel, sr=18000, n_fft=1024, hop_length=512,
                                                       n_mels=128)

            combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 时间帧数)
            mel_features.append(combined_mel)

        audio_input = np.array(mel_features)
        audio_input = torch.from_numpy(audio_input).to(self.device)
        # visual_input = torch.from_numpy(visual_input)
        visual_input = visual_input.permute(0, 3, 1, 2)
        audio_mask = self.audiomask(audio_input)
        audio_mask = audio_mask.view(audio_input.size(0), 2, self.y_dim, self.x_dim)
        masked_audio = audio_mask * audio_input

        audio_encode = self.audio_encoder(masked_audio)
        visual_encode = self.visual_encoder(visual_input)

        combined_encode = torch.cat((audio_encode, visual_encode), dim=1)
        combined_encode = self.fc1(combined_encode)
        av_out = self.fc2(combined_encode)

        return av_out


class MyData(Dataset):
    def __init__(self, data):
        self.data = data
        self.audio = list()
        self.visual = list()
        self.pred_tag = list()
        for data in self.data:
            self.audio.append(data['audio'][:10])
            self.visual.append(data['camera'][:10])
            self.pred_tag.append(data['rl_pred'][:10])
        self.audio = torch.from_numpy(np.array(self.audio))
        self.visual = torch.from_numpy(np.array(self.visual))
        self.pred_tag = torch.from_numpy(np.array(self.pred_tag))
        self.audio = self.audio.view(-1, *self.audio.shape[2:])
        self.visual = self.visual.view(-1, *self.visual.shape[2:])
        self.pred_tag = self.pred_tag.view(-1, *self.pred_tag.shape[2:])

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        # 我们核心需要三个数据，audio  ， img ， pred_tag ， 并且取前十个
        return self.audio[idx], self.visual[idx], self.pred_tag[idx]


def begin():
    path = "../yz/data/audio"
    files = os.listdir(path=path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = Network().to(device)

    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 将文件分为训练集和验证集，假设验证集占数据的50%
    num_files = len(files)
    val_split = int(num_files / 2)
    train_files = files[100:val_split]
    val_files = files[val_split:]

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        run_id = 0
        # 训练集

        file_path = os.path.join(path, train_files[epoch])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        dataset = MyData(data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        for batch_idx, (batch_audio, batch_visual, batch_labels) in enumerate(dataloader):
            batch_audio, batch_visual, batch_labels = batch_audio.to(device), batch_visual.to(device), batch_labels.to(
                device)

            optimizer.zero_grad()
            batch_audio = torch.zeros([16, 2, 18000])
            batch_visual = torch.zeros([16, 128, 128, 4]).to(device)
            outputs = model(batch_audio, batch_visual)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            run_id += 1

        # 在每个epoch结束时记录训练集的loss
        avg_train_loss = running_loss / run_id
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # 验证集测试部分
        model.eval()  # 设置为评估模式，关闭dropout等
        val_loss = 0.0
        val_id = 0
        with torch.no_grad():  # 不计算梯度，节省内存
            file_path = os.path.join(path, val_files[epoch])
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            dataset = MyData(data)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            for batch_idx, (batch_audio, batch_visual, batch_labels) in enumerate(dataloader):
                batch_audio, batch_visual, batch_labels = batch_audio.to(device), batch_visual.to(
                    device), batch_labels.to(device)
                batch_audio = torch.zeros([16, 2, 18000])
                batch_visual = torch.zeros([16, 128, 128, 4]).to(device)
                outputs = model(batch_audio, batch_visual)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_id += 1

        # avg_val_loss = val_loss / val_id
        avg_val_loss = val_loss
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    writer.close()  # 关闭TensorBoard的SummaryWriter


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    log_dir = './logs/visual_audio_4'
    writer = SummaryWriter(log_dir)

    begin()