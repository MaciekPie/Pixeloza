import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. ŁADOWANIE DANYCH
# ==========================================
data = torch.load("./processed_images_features.pt", map_location="cpu")
images = data["images"]


# ==========================================
# 2. POPRAWIONA ARCHITEKTURA MODELU
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=32, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)

        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embeddings.weight.t())
        )

        indices = distances.argmin(1)
        z_q = self.embeddings(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        vq_loss = (
            F.mse_loss(z_q, z_e.detach())
            + self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        )

        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss, indices.view(B, H, W)


class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),
            ResBlock(64),
            nn.Conv2d(64, 32, 3, 1, 1)
        )

        self.vq_vae = VectorQuantizer(num_embeddings=256, embedding_dim=32)

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            ResBlock(64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            ResBlock(64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq_vae(z)
        return self.decoder(z_q), vq_loss, indices


# ==========================================
# 3. TESTOWANIE I RYSOWANIE
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VQGAN().to(device)
model.load_state_dict(torch.load("./vqgan_best.pth", map_location=device))
model.eval()

original = images[:10].to(device)

with torch.no_grad():
    reconstructed, _, _ = model(original)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for i in range(10):
    # Wyświetlanie obrazów RGBA z interpolacją 'nearest' dla zachowania ostrych pikseli
    orig_img = original[i].permute(1, 2, 0).cpu().numpy()
    recon_img = reconstructed[i].permute(1, 2, 0).cpu().numpy()

    axes[0, i].imshow(orig_img, interpolation="nearest")
    axes[0, i].axis("off")

    axes[1, i].imshow(recon_img, interpolation="nearest")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original", fontsize=12, loc='left')
axes[1, 0].set_title("Rekonstrukcja VQGAN", fontsize=12, loc='left')

plt.tight_layout()
plt.show()