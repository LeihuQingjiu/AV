import os
import importlib
import librosa
import torch
import logging
import traceback
import pdb
import numpy as np
import matplotlib.pyplot as plt
from data.EgocomDataset import Dataset as ds    # 只为还原分块
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from configs.configs import load_config
from utils.logging import log_stats, save_checkpoint
from utils.losses import MaskedMSELoss
from models.modeling import AVCorrModel


def setup_environment(config):
    """初始化环境和目录"""
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    
    # 创建梅尔频谱图保存目录
    mel_plot_dir = os.path.join(config['train']['log_dir'], 'mel_plots')
    os.makedirs(mel_plot_dir, exist_ok=True)  

    # 配置日志系统
    logging.basicConfig(
        filename=os.path.join(config['train']['log_dir'], 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def prepare_dataloaders(config):
    """创建三个子集的 DataLoader"""
    Dataset = importlib.import_module(config['data']['dataset_module']).Dataset
    # 为每个子集创建独立数据集
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = Dataset(config, split=split)
        if split == "train":
            total_samples = len(dataset)
            subset_size = total_samples // 1  # 直接取整数部分
            dataset = Subset(dataset, indices=range(subset_size))
            print(f"[DEBUG] 使用部分训练数据: {subset_size}/{total_samples}")
            
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'] if split == 'train' else config['val']['batch_size'],
            shuffle=(split == 'train'),  # 仅训练集打乱
            num_workers=config['train']['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
        dataloaders[split] = dataloader
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def plot_melspectrograms(original, generated, sample_rate, hop_length, n_mels, 
                        titles=('Original', 'Generated'), figsize=(15, 6)):
    """
    改进版梅尔频谱对比可视化
    参数:
        original (torch.Tensor): 原始梅尔频谱 (batch, channels, n_mels, time)
        generated (torch.Tensor): 生成梅尔频谱 (batch, channels, n_mels, time)
        sample_rate (int): 采样率
        hop_length (int): 帧移长度
        n_mels (int): 梅尔频带数
    """
    # 参数检查
    assert original.shape == generated.shape, "输入形状必须一致"
    batch_size, channels, _, _ = original.shape
    
    # 创建时间轴
    time_steps = original.shape[-1]
    duration = time_steps * hop_length / sample_rate
    time_axis = np.linspace(0, duration, time_steps)
    
    # 创建频率轴（梅尔频率转Hz）
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sample_rate/2)
    
    # 绘制每个样本的对比图
    figs = []
    for i in range(batch_size):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制双声道平均频谱
        for ch in range(channels):
            # 转换为分贝单位
            orig_db = librosa.power_to_db(original[i, ch].cpu().numpy(), ref=np.max)
            gen_db = librosa.power_to_db(generated[i, ch].cpu().numpy(), ref=np.max)
            
            # 原始频谱
            img_orig = librosa.display.specshow(
                orig_db, 
                x_coords=time_axis,
                y_coords=mel_freqs,
                sr=sample_rate,
                hop_length=hop_length,
                ax=ax[0],
                cmap='magma',
                vmin=-80, 
                vmax=0
            )
            
            # 生成频谱
            librosa.display.specshow(
                gen_db,
                x_coords=time_axis,
                y_coords=mel_freqs,
                sr=sample_rate,
                hop_length=hop_length,
                ax=ax[1],
                cmap='magma',
                vmin=-80,
                vmax=0
            )
        
        # 设置公共颜色条
        fig.colorbar(img_orig, ax=ax, format='%+2.0f dB')
        
        # 标签美化
        ax[0].set(title=f'{titles[0]} (Sample {i+1})', 
                xlabel='Time (s)', ylabel='Mel Frequency (Hz)')
        ax[1].set(title=f'{titles[1]} (Sample {i+1})',
                xlabel='Time (s)', ylabel='')
        
        figs.append(fig)
        plt.close(fig)
    
    return figs

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, config, writer, epoch):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        global_step = epoch * len(loader) + batch_idx
        video = batch['video'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)

        # 混合精度训练
        with torch.amp.autocast(device.type, enabled=config['train']['mix_precision']):
            pred_audio, mask = model(video, audio)
            loss = criterion(pred_audio, audio, mask)
            loss = loss / config['train']['gradient_accumulation_steps']

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
            # 梯度裁剪
            if config['train']['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])

            if batch_idx % config['train']['log_interval'] == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(
                            f'Gradients/{name}',
                            param.grad.detach(),
                            global_step
                        )
                        # writer.add_histogram(f'Parameters/{name}', param.detach().cpu(), epoch)
                    else:
                        logging.debug(f"Parameter {name} has no gradient")
            
            # 参数更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            n_show = min(config['train']['num_visual_samples'], video.size(0))
            idx = torch.randint(0, video.size(0), (n_show,))
            
            # 转换为numpy并绘制    ---------------------edit-------频谱图可视化
            idx = torch.randperm(video.size(0))[:config['train']['num_visual_samples']]
            
            def denorm(mel, config):               # 反归一化梅尔频谱
                # 假设使用数据集全局统计量
                return mel * config['data']['audio_spec']['global_std'] + config['data']['audio_spec']['global_mean']
            
            # 获取原始和生成的梅尔频谱
            orig_mel = denorm(audio[idx], config)
            gen_mel = denorm(pred_audio[idx], config)
            
            orig_mel = ds._reconstruct_mel(orig_mel)
            gen_mel = ds._reconstruct_mel(gen_mel)
            
            # 绘制对比图
            figs = plot_melspectrograms(
                orig_mel, 
                gen_mel,
                sample_rate=config['data']['audio_spec']['sample_rate'],
                hop_length=config['data']['audio_spec']['hop_length'],
                n_mels=config['data']['audio_spec']['n_mels'],
                titles=('Ground Truth', 'Predicted')
            )
            
            # 记录到TensorBoard
            for i, fig in enumerate(figs):
                writer.add_figure(
                    f'Train/Mel_Comparison/Sample_{i}',
                    fig,
                    global_step=global_step
                )
                
                # 保存高分辨率图片
                fig.savefig(
                    os.path.join(
                        config['train']['log_dir'], 
                        'mel_plots', 
                        f'train_epoch{epoch}_step{global_step}_sample{i}.png'
                    ),
                    dpi=300, 
                    bbox_inches='tight'
                )


        # 记录损失
        total_loss += loss.item()
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        progress_bar.set_postfix(loss=loss.item())



    return total_loss / len(loader)

def validate(model, loader, criterion, device, writer, global_step):
    """验证流程"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        for batch in progress_bar:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            with torch.amp.autocast(device.type, enabled=True):
                pred_audio, mask = model(video, audio)
                loss = criterion(pred_audio, audio, mask)
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # 梅尔频谱图可视化
        first_batch = next(iter(loader))
        video = first_batch['video'].to(device)
        audio = first_batch['audio'].to(device)
        pred_audio, _ = model(video, audio)

        def denorm(mel, config):
        # 假设使用数据集全局统计量
            return mel * config['data']['audio_spec']['global_std'] + config['data']['audio_spec']['global_mean']
        # 反归一化
        orig_mel = denorm(audio, config)
        gen_mel = denorm(pred_audio, config)
        
        orig_mel = ds._reconstruct_mel(orig_mel)
        gen_mel = ds._reconstruct_mel(gen_mel)
        
        # 绘制对比图
        figs = plot_melspectrograms(
            orig_mel[:config['val']['num_visual_samples']],
            gen_mel[:config['val']['num_visual_samples']],
            sample_rate=config['data']['audio_spec']['sample_rate'],
            hop_length=config['data']['audio_spec']['hop_length'],
            n_mels=config['data']['audio_spec']['n_mels'],
            titles=('Val Ground Truth', 'Val Predicted')
        )
        
        # 记录到TensorBoard
        for i, fig in enumerate(figs):
            writer.add_figure(
                f'Val/Mel_Comparison/Sample_{i}',
                fig,
                global_step=global_step
            )
            
            # 保存高分辨率图片
            fig.savefig(
                os.path.join(
                    config['train']['mel_plot_dir'], 
                    f'val_step{global_step}_sample{i}.png'
                ),
                dpi=300,
                bbox_inches='tight'
            )
    
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Val/Loss', avg_loss, global_step)
    return avg_loss

def main(config):
    # 初始化环境
    import pdb; pdb.set_trace()
    setup_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=config['train']['log_dir'])

    try:
        # 数据准备
        # train_set, val_set, test_set = prepare_dataloaders(config)
        train_loader, val_loader, test_loader = prepare_dataloaders(config)

        # 模型初始化
        model = AVCorrModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
        scaler = torch.amp.GradScaler(enabled=config['train']['mix_precision'])
        
        # 学习率调度
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config['train']['warmup_ratio'] * len(train_loader)),
            num_training_steps=len(train_loader) * config['train']['epochs']
        )
        
        criterion = MaskedMSELoss()
        best_val_loss = float('inf')

        # 训练循环
        for epoch in range(config['train']['epochs']):
            # 训练阶段
            train_loss = train_epoch(
                model, train_loader, optimizer, 
                scheduler, scaler, criterion, 
                device, config, writer, epoch
            )
            
            # 验证阶段
            val_loss = validate(
                model, val_loader, criterion,
                device, writer, epoch * len(train_loader)
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_loss': best_val_loss,
                }, filename=os.path.join(config['train']['checkpoint_dir'], 'best_model.pth'))

            # 定期保存
            if epoch % config['train']['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, filename=os.path.join(config['train']['checkpoint_dir'], f'checkpoint_{epoch}.pth'))

            # 记录学习率
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # 记录模型参数
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    # writer.add_histogram(f'Parameters/{name}', param, epoch)
                    # writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                    writer.add_histogram(f'Parameters/{name}', param.detach().cpu(), epoch)

            logging.info(f"Epoch {epoch+1}/{config['train']['epochs']} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        writer.close()
        
    except Exception as e:
        writer.close()
        traceback.print_exc()
        pdb.post_mortem()

if __name__ == "__main__":
    config = load_config('configs/base.yaml')
    main(config)