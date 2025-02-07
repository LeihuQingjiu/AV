import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, from_dim, to_dim, dim_per_head, n_head):
        super().__init__()
        out_dim = dim_per_head * n_head
        self.out_dim = out_dim
        self.dim_per_head = dim_per_head
        self.n_head = n_head

        self.from_fc = nn.Linear(from_dim, out_dim)
        self.to_fc = nn.Linear(to_dim, 2 * out_dim)
        self.ctx_fc = nn.Linear(out_dim, from_dim)

        self.ffd_fc1 = nn.Linear(from_dim, 4 * from_dim)
        self.ffd_fc2 = nn.Linear(4 * from_dim, from_dim)
        self.activation = nn.GELU()
        self.ln1 = nn.LayerNorm(from_dim)
        self.ln2 = nn.LayerNorm(from_dim)

    def forward(self, x, y, mask=None):
        B, F, _ = x.shape
        _, T, _ = y.shape
        N, D = self.n_head, self.dim_per_head

        q = self.from_fc(x)    # B, F, C
        kv = self.to_fc(y)    # B, T, C
        k, v = torch.split(kv, [N * D, N * D], dim=-1)   # B, T, C
        q = q.view((B, F, N, D))
        k = k.view((B, T, N, D))
        v = v.view((B, T, N, D))
        score = torch.einsum("bfnd,btnd->bnft", q, k)   # B, N, F, T
        score = score / D ** 0.5
        if mask is not None:
            score = score - (1.0 - mask.float()[:, None, None, :]) * 1e8
        prob = torch.softmax(score, dim=-1)   # B, N, F, T
        context = torch.einsum("bnft,btnd->bfnd", prob, v)   # B, F, N, D
        context = context.reshape((B, F, N * D))
        x = self.ln1(x + self.ctx_fc(context))

        ffd = self.ffd_fc2(self.activation(self.ffd_fc1(x)))
        x = self.ln2(x + ffd)

        return x, prob