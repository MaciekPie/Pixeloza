import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# changed to take mine dataset
class PixelArtDataset(Dataset):
    def __init__(self, images_tensor, features_tensor):
        self.images = images_tensor
        self.features = features_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx] * 2.0 - 1.0, self.features[idx]


# tutaj troche krócej i zoptymalizowane pod gpu bardziej
class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha_cumprod = torch.cumprod(1.0 - self.beta, dim=0)

    def to(self, device):
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self

    # tak samo
    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        return (
            sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise,
            noise
        )

# sinusoidalny embedding sprawdził się troche lepiej niż zwykły nn.Embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# zmienilem BatchNorm2d na GroupNorm
# groupnorm jest bardziej stabiny przy mniejszych batchach (zostawiłem batch = 128)
class SimpleBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)

    # jest taka dobra zasada że relu zbyt obcina wartości ujemne, więc lepiej używać silu lub gelu czy inne
    # wartosci ujemne warto zachować i bardzo zeskalować, poprawiają gradient nauki i stabilność nauki
    def forward(self, x, t_emb):
        h = F.silu(self.gn1(self.conv1(x)))
        t_proj = F.silu(self.time_mlp(t_emb))[:, :, None, None]
        h = h + t_proj
        return F.silu(self.gn2(self.conv2(h)))


class UNet32(nn.Module):
    def __init__(self,n_glasses,n_head,n_body,in_channels=4,time_emb_dim=128,base=64):
        super().__init__()
        self.in_channels = in_channels
        c = base

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), #zmieniony embedding na sinusoidalny
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.glasses_emb = nn.Embedding(n_glasses, time_emb_dim)
        self.head_emb = nn.Embedding(n_head, time_emb_dim)
        self.body_emb = nn.Embedding(n_body, time_emb_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 3, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # enkoder
        self.b1 = SimpleBlock(in_channels, c, time_emb_dim)
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1) # konwolucj
        self.b2 = SimpleBlock(c, 2 * c, time_emb_dim)
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)

        # wezel srodkowy
        self.mid = SimpleBlock(2 * c, 2 * c, time_emb_dim)


        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * c, c, 3, padding=1)
        )
        self.b3 = SimpleBlock(c + 2 * c, c, time_emb_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c // 2, 3, padding=1))
        self.b4 = SimpleBlock(c // 2 + c, c // 2, time_emb_dim)

        self.final_conv = nn.Conv2d(c // 2, in_channels, 1)

    def forward(self, x, t, cond, drop_cond_mask=None):
        g_e = self.glasses_emb(cond[:, 0])
        h_e = self.head_emb(cond[:, 1])
        b_e = self.body_emb(cond[:, 2])

        c_emb = torch.cat([g_e, h_e, b_e], dim=-1)

        # dodano losowe wyłączanie warunków, aby model nie był zbyt zależny od nich i lepiej generalizował, do cfg
        if drop_cond_mask is not None:
            c_emb = c_emb * (~drop_cond_mask).unsqueeze(-1)

        t_emb = self.time_mlp(t) + self.cond_mlp(c_emb)

        x1 = self.b1(x, t_emb)
        x2 = self.b2(self.down1(x1), t_emb)
        
        mid = self.mid(self.down2(x2), t_emb)

        x3 = self.b3(torch.cat([self.up2(mid), x2], dim=1),t_emb)
        x4 = self.b4(torch.cat([self.up1(x3), x1], dim=1),t_emb)

        return self.final_conv(x4)