import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 时间编码
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # 输入 t: (B,)
        return self.mlp(t[:, None].float())  # 输出 (B, dim)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.block(x)), self.block(x)  # 下采样 + 跳连

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 上采样
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=128, num_classes=10):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # 初始映射
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 编码器
        self.down1 = DownBlock(base_channels + time_emb_dim, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)

        # 中间
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )

        # 解码器
        self.up1 = UpBlock(base_channels * 8, base_channels * 2)
        self.up2 = UpBlock(base_channels * 4, base_channels)

        # 输出
        self.out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, label):
        # 处理时间与标签
        t_emb = self.time_mlp(t)            # (B, D)
        label_emb = self.label_emb(label)           # (B, D)
        cond = t_emb + label_emb                # (B, D)
        cond = cond[:, :, None, None]       # (B, D, 1, 1)
        cond = cond.expand(-1, -1, x.shape[2], x.shape[3])  # (B, D, H, W), D= 128

        x = self.conv0(x)                   # 初始卷积
        x = torch.cat([x, cond], dim=1)     # 通道维拼接条件信息 D = (x=64) + 128

        # 下采样
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)

        # 中间处理
        x_mid = self.mid(x2)

        # 上采样
        x = self.up1(x_mid, skip2)
        x = self.up2(x, skip1)
        x = self.out(x)

        return x
    
class Diffusion:
    def __init__(self, model:nn.Module, device, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = timesteps
        self.device=device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_bar)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_bar[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_bar)


    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)
        sqrt_alpha_bar = self.alphas_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_bar[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
    
    def training_losses(self, x_0, label):
        B = x_0.size(0)
        t = torch.randint(0, self.T, (B,), device=x_0.device).long().to(self.device)
        noise = torch.randn_like(x_0).to(self.device)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t, label)
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, x_t, t, label):
        pred_noise = self.model(x_t, t, label)

        alpha_t = self.alphas[t][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        one_minus_alpha = (1 - alpha_t)

        mean = (1 / sqrt_alpha_t) * (x_t - (one_minus_alpha / sqrt_one_minus_ab) * pred_noise)

        # 最后一步，不加噪声
        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            var = self.posterior_variance[t][:, None, None, None]
            return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device, label):

        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, label)
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x_t, t, t_prev, label, eta=0.0):
        # 预测噪声
        pred_noise = self.model(x_t, t, label)

        # 计算 alpha 和 alpha_bar
        alpha_t = self.alphas_bar[t][:, None, None, None]
        alpha_prev = self.alphas_bar[t_prev][:, None, None, None]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)

        # 预测 x0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / sqrt_alpha_t

        # 计算 direction
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        # sigma = 0
        dir_term = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise

        # 计算 noise term
        if eta > 0 and t[0] != 0:
            noise = torch.randn_like(x_t)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            noise_term = sigma * noise
        else:
            noise_term = 0.0

        # 计算 xt-1
        x_prev = sqrt_alpha_prev * x0_pred + dir_term + noise_term
        return x_prev

    def ddim_sample_loop(self, shape, device, label, eta=0.0, ddim_steps=100):
        x = torch.randn(shape, device=device)

        # 使用线性间隔的时间步
        step_indices = np.linspace(0, self.T - 1, ddim_steps, dtype=int)[::-1]
        step_pairs = list(zip(step_indices[:-1], step_indices[1:])) + [(step_indices[-1], 0)]  # 最后一步跳到0

        for t, t_prev in step_pairs:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            x = self.ddim_sample(x, t_batch, t_prev_batch, label, eta=eta)
        
        return x
