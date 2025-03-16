import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP



class Logger:
    def log_metrics(self, metrics_dict):
        if self.use_wandb:
            wandb.log(metrics_dict)
        if self.use_tb:
            for k, v in metrics_dict.items():
                self.writer.add_scalar(k, v, self.step)

    import logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

def log_stats(epoch, batch_idx, loss, lr):
    """
    使用 logging 模块记录训练统计信息
    :param epoch: 当前的 epoch 数
    :param batch_idx: 当前的批次索引
    :param loss: 当前批次的损失值
    :param lr: 当前的学习率
    """
    logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, Learning Rate: {lr:.6f}')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

def save_checkpoint(state, filename):
    if 'model' in state and isinstance(state['model'], torch.nn.parallel.DistributedDataParallel):
        state['model'] = state['model'].module.state_dict()
    
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

    

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']