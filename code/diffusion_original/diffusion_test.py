import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

MODEL_PATH = "./diffusion_model_original.pt"

GLASSES = 7
HEAD = 220
BODY = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_cumprod[:-1]]
        )

        self.posterior_variance = (
            self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def to(self, device):
        self.device = device
        for name in ["beta", "alpha", "alpha_cumprod", "alpha_cumprod_prev", "posterior_variance"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def add_noise(self, x_0, t):
        #Dodaje szum do oryginalnego obrazu x_0 w kroku t
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise


class SimpleBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x, t_emb):
        h = F.relu(self.bn1(self.conv1(x)))
        t_emb = F.relu(self.time_mlp(t_emb))[:, :, None, None]
        h = h + t_emb
        h = F.relu(self.bn2(self.conv2(h)))
        return h

class UNet32(nn.Module):
    """UNet warunkowy: krok czasowy + trzy ID (body, glasses, head).
    Kazde ID ma wlasna tablice embeddingow"""
    def __init__(self, n_body, n_glasses, n_head, in_channels=4, time_emb_dim=128, base=64):
        super().__init__()
        self.in_channels = in_channels
        c = base
        self.time_emb = nn.Embedding(scheduler.timesteps, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Embeddingi warunkow: kazde ID (body/glasses/head) ma wlasny wektor
        self.body_emb = nn.Embedding(n_body, time_emb_dim)
        self.glasses_emb = nn.Embedding(n_glasses, time_emb_dim)
        self.head_emb = nn.Embedding(n_head, time_emb_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Enkoder
        self.b1 = SimpleBlock(in_channels, c, time_emb_dim)        # [B, c, 32, 32]
        self.down1 = nn.MaxPool2d(2)
        self.b2 = SimpleBlock(c, 2 * c, time_emb_dim)              # [B, 2c, 16, 16]
        self.down2 = nn.MaxPool2d(2)

        # Wezel srodkowy
        self.mid = SimpleBlock(2 * c, 2 * c, time_emb_dim)         # [B, 2c, 8, 8]

        # Dekoder
        self.up2 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.b3 = SimpleBlock(3 * c, c, time_emb_dim)              # c + 2c
        self.up1 = nn.ConvTranspose2d(c, c // 2, 2, stride=2)
        self.b4 = SimpleBlock(c // 2 + c, c // 2, time_emb_dim)

        self.final_conv = nn.Conv2d(c // 2, in_channels, 1)

    def forward(self, x, t, cond):
        c_emb = self.glasses_emb(cond[:, 0]) + self.head_emb(cond[:, 1]) + self.body_emb(cond[:, 2])
        t_emb = self.time_mlp(self.time_emb(t)) + self.cond_mlp(c_emb)

        x1 = self.b1(x, t_emb)
        x2 = self.b2(self.down1(x1), t_emb)

        mid = self.mid(self.down2(x2), t_emb)

        x3 = self.b3(torch.cat([self.up2(mid), x2], dim=1), t_emb)
        x4 = self.b4(torch.cat([self.up1(x3), x1], dim=1), t_emb)

        return self.final_conv(x4)


data = torch.load("../processed_images_features.pt", map_location="cpu")
features = data["features"].long()

num_glasses = features[:, 0].max().item() + 1
num_head = features[:, 1].max().item() + 1
num_body = features[:, 2].max().item() + 1

scheduler = LinearNoiseScheduler()

model = UNet32(num_body, num_glasses, num_head).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model wczytany.")


@torch.no_grad()
def generate(body=BODY, glasses=GLASSES, head=HEAD, n=8):
    cond = torch.tensor([[glasses, head, body]] * n, device=device)
    x = torch.randn(n, 4, 32, 32, device=device)

    for i in reversed(range(1000)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        predicted_noise = model(x, t, cond)

        alpha = scheduler.alpha[i]
        alpha_cumprod = scheduler.alpha_cumprod[i]

        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * predicted_noise)

        if i > 0:
            x += torch.sqrt(scheduler.posterior_variance[i]) * torch.randn_like(x)

    return (x.clamp(-1, 1) + 1) / 2


images = generate()[:, :3].cpu().permute(0, 2, 3, 1).numpy()

fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")

plt.tight_layout()
plt.show()