import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import importlib
from torch.utils.data import DataLoader 
from models.modeling import AVCorrModel as Network
from utils.losses import HybridLoss
from utils.logging import log_stats, save_checkpoint
from transformers import get_cosine_schedule_with_warmup

def setup(rank, world_size, master_addr='localhost', master_port='1963'):
    try:
        # 设置主节点地址和端口的环境变量
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        # 初始化分布式进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        # 设置当前 CUDA 设备
        torch.cuda.set_device(rank)
        print(f"Rank {rank} initialized successfully.")
    except Exception as e:
        print(f"Error initializing rank {rank}: {e}")

def cleanup():
    try:
        # 销毁分布式进程组
        dist.destroy_process_group()
        print("Distributed process group destroyed.")
    except Exception as e:
        print(f"Error destroying process group: {e}")

def train_ddp(rank, world_size, config):
    setup(rank, world_size, config['dist']['master_addr'], config['dist']['master_port'])
    
    # 1. 准备数据
    Dataset = importlib.import_module(config['data']['dataset_module']).Dataset
    dataset = Dataset(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
    
    # 2. 初始化模型
    model = Network(config).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    scaler = torch.cuda.amp.GradScaler(enabled=config['train']['mix_precision'])
    
    for epoch in range(config['train']['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=config['train']['mix_precision']):
                output = model(batch['video'].to(rank), batch['audio'].to(rank))
                loss = HybridLoss(output, batch['audio'].to(rank))
                
            scaler.scale(loss).backward()
            
            if (batch_idx+1) % config['train']['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            # 日志记录
            if rank == 0 and batch_idx % 100 == 0:
                log_stats(epoch, batch_idx, loss.item(), optimizer.param_groups[0]['lr'])
                
        # 保存检查点
        if epoch % 10 == 0 and rank == 0:
            save_checkpoint(epoch, model, optimizer, f'checkpoint_{epoch}.pth')
    
    cleanup()

if __name__ == "__main__":
    config = {
        'dataset_module': 'EgocomDataset',
        'data_root': './data',
        'batch_size': 32,
        'lr': 2e-4,
        'epochs': 100
    }
    
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size)