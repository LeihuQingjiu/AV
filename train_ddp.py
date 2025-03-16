import os
os.environ["OMP_NUM_THREADS"] = "1"
import importlib
import logging
import torch
import traceback
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from configs.configs import load_config
from utils.logging import save_checkpoint
from utils.losses import MaskedMSELoss
from models.modeling import AVCorrModel

def setup_environment(config):
    """初始化环境和目录（仅主进程执行）"""
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(config['train']['log_dir'], 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def prepare_dataloaders(config, rank):
    """创建分布式数据加载器"""
    Dataset = importlib.import_module(config['data']['dataset_module']).Dataset
    
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = Dataset(config, split=split)
        
        # 训练集使用分布式采样器
        if split == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True
            )
            loader = DataLoader(
                dataset,
                batch_size=config['train']['batch_size'],
                sampler=sampler,
                num_workers=config['train']['num_workers'],
                pin_memory=True,
                persistent_workers=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=config['val']['batch_size'],
                shuffle=False,
                num_workers=config['val']['num_workers']
            ) if rank == 0 else None  # 非主进程不加载验证/测试集
        
        loaders[split] = loader
    print("数据处理完成")
    return loaders['train'], loaders['val'], loaders['test']

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, config, epoch, local_rank):
    """分布式训练单个epoch"""
    model.train()
    total_loss = 0.0
    
    # 仅在主进程显示进度条
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False) if local_rank == 0 else loader
    
    for batch_idx, batch in enumerate(progress_bar):
        # 数据转移到设备
        video = batch['video'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)

        # 混合精度训练
        with torch.amp.autocast(device.type, enabled=config['train']['mix_precision']):
            pred_audio, mask = model(video, audio)
            loss = criterion(pred_audio, audio, mask)
            loss = loss / config['train']['gradient_accumulation_steps']

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
            if config['train']['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # 损失汇总
        total_loss += loss.item()
    
    # 跨设备损失同步
    total_loss_tensor = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    return total_loss_tensor.item() / (len(loader) * dist.get_world_size())

def validate(model, loader, criterion, device):
    """分布式验证（仅在主进程执行）"""
    if loader is None:
        return float('inf')
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            with torch.amp.autocast(device.type):
                pred_audio, mask = model(video, audio)
                loss = criterion(pred_audio, audio, mask)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main(config):
    # 初始化分布式环境
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    import sys
    sys.stderr = open(f'stderr_rank{local_rank}.log', 'w')  # 输出所有Rank的报错信息
    sys.stdout = open(f'stdout_rank{local_rank}.log', 'w')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    global_rank = int(os.environ["RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        # rank=local_rank,
        rank=global_rank,
        world_size=world_size
    )
    device = torch.device(f"cuda:{local_rank}")

    # 主进程初始化环境
    writer = None
    if local_rank == 0:
        setup_environment(config)
        writer = SummaryWriter(log_dir=config['train']['log_dir'])

    try:
        # 准备数据加载器
        train_loader, val_loader, _ = prepare_dataloaders(config, local_rank)
        
        # 初始化模型
        model = AVCorrModel(config).to(device)
        model = DDP(model, device_ids=[local_rank])
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['train']['lr'] * dist.get_world_size()  # 学习率线性缩放
        )
        scaler = torch.amp.GradScaler(enabled=config['train']['mix_precision'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config['train']['warmup_ratio'] * len(train_loader)),
            num_training_steps=len(train_loader) * config['train']['epochs']
        )
        criterion = MaskedMSELoss().to(device)

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(config['train']['epochs']):
            # 设置分布式采样器epoch
            if train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch)
            
            # 训练阶段
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, criterion, device, config, epoch, local_rank
            )

            # 验证阶段（仅主进程）
            val_loss = float('inf')
            if local_rank == 0:
                val_loss = validate(model.module, val_loader, criterion, device)
                writer.add_scalar('Val/Loss', val_loss, epoch)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                    }, filename=os.path.join(config['train']['checkpoint_dir'], 'best_model.pth'))

                # 记录日志
                logging.info(f"Epoch {epoch+1}/{config['train']['epochs']} | "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 进程同步
            dist.barrier()

        if local_rank == 0:
            writer.close()

    except Exception as e:
        if local_rank == 0:
            logging.error(f"训练异常终止: {str(e)}")
            traceback.print_exc()
        raise

if __name__ == "__main__":
    config = load_config('configs/base.yaml')
    main(config)