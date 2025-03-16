import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim = MS_SSIM(data_range=1.0, channel=1)
        self.alpha = alpha

    def forward(self, pred, target, mask):
        # 计算L1损失
        import pdb; pdb.set_trace()
        l1 = self.l1_loss(pred[mask], target[mask])
        
        # 计算MS-SSIM损失（灰度化处理）
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        ssim_loss = 1 - self.ssim(pred_gray[mask], target_gray[mask])
        
        return self.alpha * l1 + (1 - self.alpha) * ssim_loss
    
# class MaskedMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss(reduction='none')

#     def forward(self, pred, target, mask):
#         loss = self.mse(pred, target)
#         mask = mask.unsqueeze(-1)
#         print(f"pred shape: {pred.shape}, target shape: {target.shape}, mask shape: {mask.shape}")
#         masked_loss = (loss * mask).sum() / mask.sum()  # 论文公式(1)
#         return masked_loss

# class MaskedMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss(reduction='none')

#     def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         # 提取被掩码的目标token
#         target_masked = target[:, mask.squeeze(), :]  # [1, S, 32]
        
#         # 计算逐元素MSE损失
#         loss = self.mse(pred, target_masked)  # [1, S, 32]
        
#         mask_matched = torch.ones_like(loss, dtype=torch.bool)  # [1, S, 32]
        
#         masked_loss = (loss * mask_matched).sum() / mask_matched.sum()
#         return masked_loss
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # [16,784,32]
        loss = self.mse(pred, target)  # [16,784,32]

        # 3. 仅保留被掩码位置的损失
        masked_loss = loss[mask_expanded]  # [num_masked_tokens,]

        # 4. 计算平均损失
        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        return masked_loss.mean()